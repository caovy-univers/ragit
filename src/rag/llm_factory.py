from typing import Any, Dict, Literal, Optional, Union
import os

import torch
from loguru import logger
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

Backend = Literal["ollama", "phogpt"]


def _prefer_torch_attention(config) -> None:
    """
    Force safer attention implementation to avoid FlashAttention/Triton paths.

    For MPT-based models (e.g., PhoGPT), configs often expose either:
      - config.attn_impl: str
      - config.attn_config: dict with key 'attn_impl'

    We set both (when present) to 'torch'. We also set a few env hints that
    some repos check before trying flash attention backends.
    """
    if hasattr(config, "attn_impl"):
        try:
            config.attn_impl = "torch"
        except Exception:
            pass

    attn_conf = getattr(config, "attn_config", None)
    if isinstance(attn_conf, dict):
        attn_conf["attn_impl"] = "torch"

    # Env hints (harmless if ignored)
    os.environ.setdefault("MPT_ATTENTION_IMPL", "torch")
    os.environ.setdefault("PHOGPT_NO_FLASH", "1")
    os.environ.setdefault("DISABLE_FLASH_ATTN", "1")


def _build_phogpt_pipeline(
    model_name: str,
    device: Optional[Union[int, str]] = None,
    use_4bit: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> HuggingFacePipeline:
    """
    Build a Hugging Face text-generation pipeline for PhoGPT.

    Robust loading plan (no dynamic modules):
      - Load AutoConfig with trust_remote_code=False (built-in Transformers).
      - Force 'torch' attention to avoid flash/triton deps.
      - Try AutoModelForCausalLM (standard generation).
      - Fallback to AutoModel exactly as requested by the user
        (dtype='auto') to remain compatible.

    This avoids ImportErrors for optional GPU-only dependencies (flash-attn, triton).
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # 1) Load config via Transformers built-ins (NO remote code)
    logger.info(f"Preparing config (no remote code) for {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    _prefer_torch_attention(config)

    # 2) Precision / quantization
    load_kwargs: Dict[str, Any] = dict(trust_remote_code=False, config=config)
    if use_4bit:
        load_kwargs.update(
            dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                device_map="auto",
            )
        )
    else:
        load_kwargs.update(
            dict(
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        )

    # 3) Tokenizer (no remote code needed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

    # 4) Try standard CausalLM first
    model = None
    try:
        logger.info(f"Loading PhoGPT (built-in) as AutoModelForCausalLM: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        logger.warning(f"AutoModelForCausalLM failed ({e}). Falling back to AutoModel.")
        # Fallback: EXACT pattern you requested
        fallback_kwargs = dict(load_kwargs)
        if not use_4bit:
            fallback_kwargs["dtype"] = "auto"
        logger.info(f"Loading PhoGPT (built-in) as AutoModel: {model_name}")
        model = AutoModel.from_pretrained(model_name, **fallback_kwargs)

    # 5) Safe pad token id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=temperature > 0.0,
        pad_token_id=pad_id,
    )
    return HuggingFacePipeline(pipeline=gen)


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
    """
    if backend == "ollama":
        return OllamaLLM(model=model_name)

    if backend == "phogpt":
        return _build_phogpt_pipeline(
            model_name=model_name,
            device=device,
            use_4bit=use_4bit,
            **gen_kwargs,
        )

    raise ValueError(f"Unknown backend: {backend}")
