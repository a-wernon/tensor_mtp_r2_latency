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
from transformers.cache_utils import DynamicCache

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


@torch.inference_mode()
def time_target_cached(
    model,
    prefix_lens: list[int],
    draft_ks: list[int],
    iters: int,
    warmup: int,
    device: str,
) -> dict[str, dict[str, dict]]:
    """Time a 'verification-style' forward: k query tokens on a cached prefix of length L.

    Measures ONLY the model forward — cache construction is done outside the
    timed region on each iteration. The earlier version put `make_cache()`
    inside the timed callable, which meant the timing picked up ~30 ms of
    DynamicCache rebuild overhead and the resulting numbers were flat in
    L and k. By constructing the cache in the outer loop and only recording
    CUDA events around `model(...)`, we now report the true forward cost.

    Implementation notes:
      - A full forward over a length-L prefix primes the reference cache
        and gives us a snapshot `base_kv` of per-layer (K, V) tensor refs.
        We never mutate these tensors — the model's forward uses torch.cat
        to produce fresh layer tensors, leaving base_kv intact across iters.
      - Each iteration builds a fresh DynamicCache via the
        `ddp_cache_data=base_kv` constructor OUTSIDE the timed window, then
        CUDA events wrap only the `model(...)` call.
      - `position_ids` and `cache_position` are passed explicitly; Qwen3
        accepts both.

    Returns a nested dict: results[str(L)][str(k)] -> summary_stats.
    """
    results: dict[str, dict[str, dict]] = {}
    V = model.config.vocab_size

    for L in prefix_lens:
        results[str(L)] = {}
        logger.info(f"[target-cached] priming KV cache at L={L}")
        prefix_ids = torch.randint(0, V, (1, L), device=device)
        out_prefix = model(
            prefix_ids,
            use_cache=True,
            past_key_values=DynamicCache(),
        )
        prefix_cache = out_prefix.past_key_values
        # Snapshot per-layer K/V tensor refs. Supports both the newer
        # `.layers` API (each layer holds .keys/.values) and the older
        # flat-list API (.key_cache / .value_cache).
        if hasattr(prefix_cache, "layers"):
            base_kv = [(layer.keys, layer.values) for layer in prefix_cache.layers]
        else:
            base_kv = list(zip(prefix_cache.key_cache, prefix_cache.value_cache))

        def build_cache() -> DynamicCache:
            return DynamicCache(ddp_cache_data=base_kv)

        for k in draft_ks:
            draft_ids = torch.randint(0, V, (1, k), device=device)
            pos = torch.arange(L, L + k, device=device).unsqueeze(0)
            cache_pos = torch.arange(L, L + k, device=device)

            # Warmup — cache construction included here too; we don't care
            # about absolute warmup time, only that CUDA kernels are warm.
            for _ in range(warmup):
                cache = build_cache()
                _ = model(
                    input_ids=draft_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=pos,
                    cache_position=cache_pos,
                )
            torch.cuda.synchronize()

            # Timed loop: cache construction outside, CUDA events around the
            # model forward only.
            times: list[float] = []
            for _ in range(iters):
                cache = build_cache()
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(
                    input_ids=draft_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=pos,
                    cache_position=cache_pos,
                )
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))

            stats = summary_stats(times)
            logger.info(
                f"[target-cached] L={L} k={k}: median={stats['median_ms']:.3f}ms  "
                f"mean={stats['mean_ms']:.3f}±{stats['stdev_ms']:.3f}ms  "
                f"min={stats['min_ms']:.3f}ms   (forward only)"
            )
            results[str(L)][str(k)] = stats
            del draft_ids

        del prefix_cache, out_prefix, base_kv, prefix_ids
        torch.cuda.empty_cache()

    return results
