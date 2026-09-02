"""Model loading helpers for multi-precision (FP16 / INT8 / INT4_NF4) runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Precision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4_NF4 = "int4_nf4"


class ModelSize(str, Enum):
    GEMMA2_2B = "2b"
    GEMMA2_9B = "9b"
    NEMOTRON_MINI_4B = "nemotron-mini-4b"


MODEL_IDS: dict[ModelSize, str] = {
    ModelSize.GEMMA2_2B: "google/gemma-2-2b-it",
    ModelSize.GEMMA2_9B: "google/gemma-2-9b-it",
    ModelSize.NEMOTRON_MINI_4B: "nvidia/Nemotron-Mini-4B-Instruct",
}

# Small local models (~2–4B) share the same VRAM heuristics.
_LOCAL_SMALL_MODELS = frozenset({ModelSize.GEMMA2_2B, ModelSize.NEMOTRON_MINI_4B})


@dataclass(frozen=True)
class LoadConfig:
    model_size: ModelSize
    precision: Precision
    device_map: str = "auto"
    trust_remote_code: bool = False

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.model_size]


def recommended_precisions(model_size: ModelSize, vram_gb: float) -> list[Precision]:
    """Heuristic precision menu for local 16GB vs larger cloud GPUs."""
    if model_size in _LOCAL_SMALL_MODELS:
        if vram_gb >= 12:
            return [Precision.FP16, Precision.INT8, Precision.INT4_NF4]
        return [Precision.INT8, Precision.INT4_NF4]
    # 9B: prefer quantized on consumer 16GB cards
    if vram_gb >= 40:
        return [Precision.FP16, Precision.INT8, Precision.INT4_NF4]
    if vram_gb >= 16:
        return [Precision.INT8, Precision.INT4_NF4]
    return [Precision.INT4_NF4]


def build_quantization_config(precision: Precision) -> Any | None:
    """Return a BitsAndBytesConfig (or None for FP16). Imports are lazy."""
    if precision == Precision.FP16:
        return None

    from transformers import BitsAndBytesConfig
    import torch

    if precision == Precision.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)

    if precision == Precision.INT4_NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    raise ValueError(f"Unsupported precision: {precision}")


def load_model_and_tokenizer(config: LoadConfig) -> tuple[Any, Any]:
    """
    Load tokenizer + causal LM for the requested precision.

    Skeleton note: call this from the profiler once CUDA deps are installed.
    FP16 for 9B may OOM on 16GB; use recommended_precisions() to gate runs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quantization_config(config.precision)
    dtype = torch.float16 if config.precision == Precision.FP16 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quant_config,
        device_map=config.device_map,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return model, tokenizer
