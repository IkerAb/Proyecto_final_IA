"""Carga del modelo (base o base + adaptador LoRA) y generación de SQL.

- CUDA: bf16/fp16 (opcionalmente 4-bit con bitsandbytes para evaluar en GPUs pequeñas).
- Apple Silicon (MPS): fp16.
- CPU: bf16 por defecto para caber en 8 GB de RAM (SQLOCAL_CPU_DTYPE=float32 para más precisión).
  bitsandbytes 4-bit NO se usa en CPU: el adaptador se fusiona con el modelo (merge_and_unload).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .config import ADAPTER_DIR, BASE_MODEL_ID, MAX_NEW_TOKENS, adapter_available
from .prompting import build_messages, extract_sql


@dataclass
class Generation:
    sql: str
    raw: str
    latency_s: float


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return getattr(torch, os.environ.get("SQLOCAL_CPU_DTYPE", "bfloat16"))


class SQLModel:
    def __init__(self, variant: str = "base", device: str | None = None, load_in_4bit: bool = False):
        if variant not in ("base", "finetuned"):
            raise ValueError(f"Variante desconocida: {variant}")
        if variant == "finetuned" and not adapter_available():
            raise FileNotFoundError(
                f"No hay adaptador LoRA en {ADAPTER_DIR}. Entrena primero con scripts/train.py."
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.variant = variant
        self.device = device or pick_device()
        dtype = _dtype_for(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        kwargs = {"torch_dtype": dtype}
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)

        if variant == "finetuned":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
            if not load_in_4bit:
                model = model.merge_and_unload()

        if "device_map" not in kwargs:
            model = model.to(self.device)
        model.eval()
        self.model = model
        self.dtype = str(dtype).replace("torch.", "")
        self._torch = torch

    @property
    def info(self) -> dict:
        return {"variant": self.variant, "base_model": BASE_MODEL_ID, "device": self.device, "dtype": self.dtype}

    def generate(self, question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
        prompt = self.tokenizer.apply_chat_template(
            build_messages(question), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        start = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return Generation(sql=extract_sql(raw), raw=raw, latency_s=latency)
