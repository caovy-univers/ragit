import logging
import sys
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Literal, Optional, Union

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:  # pragma: no cover – optional dependency
    _HAS_BNB = False

Backend = Literal["ollama", "phogpt"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _select_dtype() -> torch.dtype:
    """Select an efficient dtype depending on GPU capability.

    Returns:
        torch.dtype: bfloat16 on Ampere+ GPUs, else float16 on CUDA, else float32.
    """
    if torch.cuda.is_available():
        major_cap = torch.cuda.get_device_capability(0)[0]
        if major_cap >= 8:  # Ampere or newer: prefer bf16
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _build_phogpt_pipeline(
    model_name_or_path: str,
    device: Optional[str] = None,
    use_4bit: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.10,
    do_sample: Optional[bool] = None,
) -> HuggingFacePipeline:
    """Create a LangChain `HuggingFacePipeline` for PhoGPT(-Chat).

    Notes:
        - Uses `return_full_text=False` to prevent the prompt from being echoed.
        - Works with either a HF repo id (e.g., "vinai/PhoGPT-4B-Chat") or a local path.

    Args:
        model_name_or_path (str): HF repo id or local folder.
        device (Optional[str]): "cuda" | "cpu". Auto-detected if None.
        use_4bit (bool): Enable 4-bit quantization via BitsAndBytes (if available).
        max_new_tokens (int): Maximum number of generated tokens.
        temperature (float): Sampling temperature. Set > 0 to enable sampling.
        top_p (float): Nucleus sampling probability mass.
        top_k (int): Top-k sampling filter.
        repetition_penalty (float): Penalty to discourage repetition (>= 1.0).
        do_sample (Optional[bool]): Force sampling on/off; if None, inferred from other params.

    Returns:
        HuggingFacePipeline: LangChain-compatible LLM wrapper.
    """
    is_cuda = torch.cuda.is_available()
    if device is None:
        device = "cuda" if is_cuda else "cpu"

    torch_dtype = _select_dtype()

    quant_config: Optional["BitsAndBytesConfig"] = None
    device_map: Optional[Union[str, dict]] = None

    if use_4bit and _HAS_BNB:
        # Enable 4-bit quantization for large models on consumer GPUs.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"
    elif is_cuda:
        # Let HF dispatch to the available GPU automatically when possible.
        device_map = "auto"

    logger.info(
        f"Loading model='{model_name_or_path}' (device={device}, 4bit={use_4bit and _HAS_BNB}, dtype={torch_dtype})"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_config,
    )

    # Infer sampling behavior if not explicitly provided
    if do_sample is None:
        do_sample = (temperature and temperature > 0.0) or (top_p and top_p < 1.0) or (top_k and top_k > 0)

    # Deterministic-by-default generation pipeline suitable for RAG.
    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        max_new_tokens=int(max_new_tokens),
        return_full_text=False,  # IMPORTANT: do not echo the prompt
    )

    return HuggingFacePipeline(pipeline=gen_pipe)


def build_llm(
    backend: Backend,
    model_name: str,
    device: Optional[str] = None,
    use_4bit: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.10,
    do_sample: Optional[bool] = None,
) -> Union[HuggingFacePipeline, OllamaLLM]:
    """Factory: build the requested LLM backend.

    Args:
        backend (Literal["ollama", "phogpt"]): LLM backend identifier.
        model_name (str): HF repo id, local path (PhoGPT), or Ollama model name.
        device (Optional[str]): Device hint ("cuda" or "cpu"). Auto if None.
        use_4bit (bool): Enable 4-bit quantization (PhoGPT only, requires bitsandbytes).
        max_new_tokens (int): Generation length limit.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability mass.
        top_k (int): Top-k sampling filter.
        repetition_penalty (float): Repetition penalty (>= 1.0).
        do_sample (Optional[bool]): Force sampling on/off; if None, inferred.

    Returns:
        Union[HuggingFacePipeline, OllamaLLM]: LangChain LLM instance.
    """
    if backend == "ollama":
        return OllamaLLM(model=model_name)

    if backend == "phogpt":
        return _build_phogpt_pipeline(
            model_name_or_path=model_name,
            device=device,
            use_4bit=use_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    raise ValueError(f"Unknown backend: {backend}")
