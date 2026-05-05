"""Target forward-pass timing on a frozen causal-LM (Qwen3 family by default).

We time a *full* (non-cached) forward pass at several sequence lengths. This
gives a fixed denominator for the t_CP / t_target ratio. The exact cost of
verification in a deployed spec-decode loop depends on whether the target
uses a KV cache, the draft length, etc.; we report both a pure-forward
number and a "decode-sized" number (small seq_len with no cache) so the
ratio can be interpreted at either end of the range.
"""

from __future__ import annotations

import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from .utils import cuda_time_ms, summary_stats


def load_target(model_name: str, dtype: torch.dtype, device: str):
    logger.info(f"Loading target: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=device
    ).eval()
    H = model.config.hidden_size
    V = model.config.vocab_size
    logger.info(f"Target loaded: hidden_size={H} vocab_size={V}")
    return model


@torch.inference_mode()
def time_target(
    model,
    seq_lens: list[int],
    iters: int,
    warmup: int,
    device: str,
) -> dict[int, dict]:
    """Time a single forward pass at each seq_len (no KV cache)."""
    results: dict[int, dict] = {}
    V = model.config.vocab_size
    for L in seq_lens:
        input_ids = torch.randint(0, V, (1, L), device=device)

        def fn() -> None:
            _ = model(input_ids, use_cache=False)

        logger.info(f"[target] timing forward at seq_len={L}")
        times = cuda_time_ms(fn, iters=iters, warmup=warmup)
        stats = summary_stats(times)
        logger.info(
            f"  seq_len={L}: median={stats['median_ms']:.2f}ms  "
            f"mean={stats['mean_ms']:.2f}±{stats['stdev_ms']:.2f}ms  "
            f"min={stats['min_ms']:.2f}ms"
        )
        results[L] = stats
        del input_ids
    torch.cuda.empty_cache()
    return results
