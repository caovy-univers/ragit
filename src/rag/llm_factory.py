import os
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any, Literal, Optional, Union

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

Backend = Literal["ollama", "phogpt"]


def _resolve_device(device: str | None) -> str:
    """
    Resolve the target device.

    Args:
        device: One of {"cuda", "cpu", "auto"} or None. If None/"auto",
            returns "cuda" when a CUDA GPU is available, otherwise "cpu".

    Returns:
        A normalized device string: "cuda" or "cpu".
    """
    if device in (None, "auto"):
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device

def _build_phogpt_pipeline(
    model_name_or_path: str,
    *,
    device: str | None = None,
    use_4bit: bool = False,
    max_new_tokens: int = 256,
):
    """
    Build a text-generation HF pipeline for PhoGPT.

    Args:
        model_name_or_path (str): HF id or local path (e.g. 'models/PhoGPT-4B').
        device (str|None): 'cuda', 'cpu', or None to auto-select.
        use_4bit (bool): Enable bitsandbytes 4-bit quantization when available.
        max_new_tokens (int): Generation cap for speed and memory safety.

    Returns:
        HuggingFacePipeline: LangChain-compatible LLM wrapper.
    """
    from langchain_community.llms import HuggingFacePipeline

    is_cuda = torch.cuda.is_available()
    if device is None:
        device = "cuda" if is_cuda else "cpu"

    prefer_bf16 = is_cuda and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if prefer_bf16 else (torch.float16 if is_cuda else torch.float32)

    quant_config = None
    device_map = None
    dtype_arg = dtype

    if use_4bit:
        # 4-bit quantization: accelerate gère les devices → pas de device= dans pipeline()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if is_cuda else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
        dtype_arg = None  # dtype contrôlé par bitsandbytes
        logger.info(f"[PhoGPT] Loading model='{model_name_or_path}' on device={device}, dtype=4-bit")
    else:
        # non-quantized: on peut fixer device directement
        device_map = 0 if device == "cuda" else None
        logger.info(f"[PhoGPT] Loading model='{model_name_or_path}' on device={device}, dtype={dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype_arg,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Arguments communs à la pipeline (greedy = pas de multinomial → évite les NaN/Inf CUDA)
    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.05,
        max_new_tokens=max_new_tokens,
    )

    # ⚠️ NE PAS passer `device` si device_map="auto" (accelerate)
    if device_map != "auto":
        pipe_kwargs["device"] = 0 if device == "cuda" else -1

    hf_pipe = pipeline(**pipe_kwargs)

    if is_cuda:
        free = torch.cuda.mem_get_info()[0] / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"[PhoGPT] GPU memory free={free:.2f} GB / total={total:.2f} GB")

    return HuggingFacePipeline(pipeline=hf_pipe)

def build_llm(
    backend: Backend,
    model_name: str,
    *,
    device: Optional[Union[int, str]] = None,
    use_4bit: bool = False,
    **gen_kwargs: Any,
) -> Union[OllamaLLM, HuggingFacePipeline]:
    """
    Factory: return a LangChain LLM wrapper matching the chosen backend.

    - 'ollama' => OllamaLLM (simple and robust)
    - 'phogpt' => HuggingFacePipeline with a safe, portable loading path
                  (no remote code, torch attention).

    Args:
        backend: The backend to use ('ollama' or 'phogpt').
        model_name: Model name or path
        device: Device to use ('auto', 'cpu', 'cuda', or 'cuda:<index>').
                Only used for 'phogpt'. If 'auto', picks GPU if available, else
                CPU. If None, defaults to 'auto'.
        use_4bit: Whether to load the model in 4-bit mode
        **gen_kwargs: Additional generation kwargs (only for 'phogpt').*

    Returns:
        An instance of the selected LLM wrapper.
    """
    if backend == "ollama":
        return OllamaLLM(model=model_name)

    if backend == "phogpt":
        return _build_phogpt_pipeline(
            model_name_or_path=model_name,
            device=device,
            use_4bit=use_4bit,
            max_new_tokens=256,
        )

    raise ValueError(f"Unknown backend: {backend}")
