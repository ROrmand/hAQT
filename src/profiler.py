"""CUDA + NVML profiling for VRAM, latency, and throughput."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from src.quantizer import LoadConfig, ModelSize, Precision


@dataclass
class ProfileMetrics:
    model_size: str
    precision: str
    peak_vram_mb: float | None
    ttft_ms: float | None
    tokens_per_sec: float | None
    total_latency_ms: float | None
    num_tokens: int = 0
    notes: list[str] = field(default_factory=list)
    task_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileResult:
    """Metrics plus decoded generations for scoring."""

    metrics: ProfileMetrics
    generations: list[str] = field(default_factory=list)


class GpuMemoryProbe:
    """Thin wrapper around NVML (``nvidia-ml-py`` / ``import pynvml``) with safe fallbacks."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._handle = None
        self._nvml = None
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._nvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def used_mb(self) -> float | None:
        if not self._nvml or not self._handle:
            return None
        info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / (1024 ** 2)

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def _cuda_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def release_cuda() -> None:
    """Force GC + empty the CUDA cache between matrix cells."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_prompt(tokenizer: Any, prompt: str, device: Any) -> dict[str, Any]:
    """Prefer chat template for instruct checkpoints; fall back to plain encode."""
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                encoded = {"input_ids": encoded}
            return {k: v.to(device) for k, v in encoded.items()}
        except Exception:
            pass

    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def measure_generation(
    generate_fn: Callable[[], tuple[int, float | None]],
    *,
    probe: GpuMemoryProbe | None = None,
) -> tuple[float, float | None, int, float | None]:
    """
    Time a generation call.

    generate_fn must return (num_new_tokens, ttft_ms_or_None).
    Returns: total_latency_ms, ttft_ms, num_tokens, peak_vram_mb
    """
    probe = probe or GpuMemoryProbe()
    peak = probe.used_mb()

    _cuda_synchronize()
    t0 = time.perf_counter()
    num_tokens, ttft_ms = generate_fn()
    _cuda_synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    after = probe.used_mb()
    if peak is not None and after is not None:
        peak = max(peak, after)
    elif after is not None:
        peak = after

    return total_ms, ttft_ms, num_tokens, peak


def _failed_result(config: LoadConfig, notes: list[str]) -> ProfileResult:
    return ProfileResult(
        metrics=ProfileMetrics(
            model_size=config.model_size.value,
            precision=config.precision.value,
            peak_vram_mb=None,
            ttft_ms=None,
            tokens_per_sec=None,
            total_latency_ms=None,
            notes=notes,
        ),
        generations=[],
    )


def profile_prompt_batch(
    config: LoadConfig,
    prompts: list[str],
    *,
    max_new_tokens: int = 64,
) -> ProfileResult:
    """
    Load model at given precision, generate on prompts, profile, then free VRAM.

    If CUDA/model weights are unavailable, return a metrics shell with notes
    instead of crashing.
    """
    notes: list[str] = []
    try:
        import torch
        from src.quantizer import load_model_and_tokenizer
    except Exception as exc:  # pragma: no cover - env dependent
        return _failed_result(config, [f"Import failed: {exc}"])

    if not torch.cuda.is_available() and config.precision != Precision.FP16:
        notes.append("CUDA not available; quantized profiling skipped.")
        return _failed_result(config, notes)

    # Clear leftover allocations from a prior matrix cell before measuring.
    release_cuda()
    probe = GpuMemoryProbe()
    baseline_vram = probe.used_mb()

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as exc:
        release_cuda()
        probe.close()
        return _failed_result(config, [f"Model load failed: {exc}"])

    total_tokens = 0
    total_ms = 0.0
    first_ttft: float | None = None
    peak_vram: float | None = probe.used_mb()
    generations: list[str] = []
    device = _model_device(model)

    try:
        for prompt in prompts:
            encoded = _encode_prompt(tokenizer, prompt, device)
            prompt_len = int(encoded["input_ids"].shape[-1])
            decoded_holder: list[str] = []

            def _gen(
                encoded=encoded,
                prompt_len=prompt_len,
                decoded_holder=decoded_holder,
            ) -> tuple[int, float | None]:
                t_start = time.perf_counter()
                # Single-shot generate; TTFT approximated as full decode until
                # streaming callbacks land in a later increment.
                with torch.inference_mode():
                    out = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                new_tokens = int(out.shape[-1] - prompt_len)
                text = tokenizer.decode(
                    out[0, prompt_len:],
                    skip_special_tokens=True,
                ).strip()
                decoded_holder.append(text)
                return new_tokens, elapsed_ms

            batch_ms, ttft_ms, n_tok, vram = measure_generation(_gen, probe=probe)
            generations.append(decoded_holder[0] if decoded_holder else "")
            total_ms += batch_ms
            total_tokens += n_tok
            if first_ttft is None:
                first_ttft = ttft_ms
            if vram is not None:
                peak_vram = vram if peak_vram is None else max(peak_vram, vram)
    except Exception as exc:
        notes.append(f"Generation failed: {exc}")
        generations = []
    finally:
        # Drop caller refs before emptying the cache (del inside a helper is a no-op).
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        model = None
        tokenizer = None
        release_cuda()
        probe.close()

    # Report peak VRAM attributable to this cell (above pre-load baseline).
    if peak_vram is not None and baseline_vram is not None:
        peak_vram = max(0.0, peak_vram - baseline_vram)

    tps = (total_tokens / (total_ms / 1000.0)) if total_ms > 0 else None
    return ProfileResult(
        metrics=ProfileMetrics(
            model_size=config.model_size.value,
            precision=config.precision.value,
            peak_vram_mb=peak_vram,
            ttft_ms=first_ttft,
            tokens_per_sec=tps,
            total_latency_ms=total_ms if total_ms > 0 else None,
            num_tokens=total_tokens,
            notes=notes,
        ),
        generations=generations,
    )


def empty_matrix(
    model_sizes: list[ModelSize],
    precisions: list[Precision],
) -> list[ProfileMetrics]:
    """Placeholder rows for the dashboard before real runs exist."""
    return [
        ProfileMetrics(
            model_size=m.value,
            precision=p.value,
            peak_vram_mb=None,
            ttft_ms=None,
            tokens_per_sec=None,
            total_latency_ms=None,
            notes=["Not run yet"],
        )
        for m in model_sizes
        for p in precisions
    ]
