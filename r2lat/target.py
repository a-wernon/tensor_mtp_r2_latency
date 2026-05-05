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

    This approximates what the target actually does in spec-decode verification
    — process a small batch of draft tokens against an already-populated KV
    cache. It is almost always MUCH cheaper than a full forward at prefix_len
    without cache.

    Implementation notes:
      - We prime a DynamicCache once per L by running a single full forward
        over a random prefix of length L.
      - Between timed iterations we restore the cache to its pre-forward
        state by reassigning `key_cache` / `value_cache` lists to their
        snapshotted references. The target's forward creates new tensors
        (torch.cat) rather than mutating in place, so the snapshots stay
        valid across iterations — no deep copy needed.
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
        # Snapshot the per-layer K/V tensor refs. The forward never mutates
        # them in place (it uses torch.cat to produce fresh tensors), so
        # keeping these refs is enough to reset state between iterations.
        base_keys = list(prefix_cache.key_cache)
        base_values = list(prefix_cache.value_cache)

        def make_cache() -> DynamicCache:
            c = DynamicCache()
            c.key_cache = list(base_keys)
            c.value_cache = list(base_values)
            # HF keeps a token counter on some versions; make sure it agrees.
            if hasattr(c, "_seen_tokens"):
                c._seen_tokens = L
            return c

        for k in draft_ks:
            draft_ids = torch.randint(0, V, (1, k), device=device)
            pos = torch.arange(L, L + k, device=device).unsqueeze(0)
            cache_pos = torch.arange(L, L + k, device=device)

            def fn() -> None:
                cache = make_cache()
                _ = model(
                    input_ids=draft_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=pos,
                    cache_position=cache_pos,
                )

            logger.info(f"[target-cached] timing L={L} k={k}")
            times = cuda_time_ms(fn, iters=iters, warmup=warmup)
            stats = summary_stats(times)
            logger.info(
                f"  L={L} k={k}: median={stats['median_ms']:.3f}ms  "
                f"mean={stats['mean_ms']:.3f}±{stats['stdev_ms']:.3f}ms"
            )
            results[str(L)][str(k)] = stats
            del draft_ids

        del prefix_cache, out_prefix, base_keys, base_values, prefix_ids
        torch.cuda.empty_cache()

    return results
