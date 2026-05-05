"""R2 entry point — CP sampling latency microbenchmark.

Usage:
    python run_latency.py --config configs/latency.yaml
    python run_latency.py --config configs/latency.yaml --smoke
    # reuse cached target-forward timings:
    python run_latency.py --config configs/latency.yaml --skip-target

Outputs under runs/r2/<run_name>_<ts>/:
    config.json, run.log, summary.json, cp_latency.png, ratio_grid.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from r2lat.cp import FFSampler, SharedTrunkCPSampler
from r2lat.target import load_target, time_target
from r2lat.utils import (
    configure_logger,
    cuda_time_ms,
    dtype_from_str,
    dump_json,
    env_threads,
    load_yaml,
    make_run_dir,
    set_seed,
    summary_stats,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity pass: reduced grid + fewer iters (~1 min).",
    )
    p.add_argument(
        "--skip-target",
        action="store_true",
        help="Do not load the target. Reuse `target_cache_path` from the config.",
    )
    return p.parse_args()


def _pick_trunk_bytes_budget(V: int, H: int, dtype: torch.dtype) -> int:
    bytes_per = {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}[dtype]
    return V * H * bytes_per


def _log_p_tensor_bytes(n: int, r: int, V: int) -> int:
    # log_p in float32 after F.log_softmax(.float(), ...)
    return n * r * V * 4


def _bench_ff(
    H: int,
    V: int,
    n: int,
    W_LM: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> dict:
    ff = FFSampler(H, n, V, W_LM, device, dtype)
    h = torch.randn(n, H, device=device, dtype=dtype)
    times = cuda_time_ms(lambda: ff.sample(h), iters=iters, warmup=warmup)
    del ff, h
    return summary_stats(times)


def _bench_cp(
    H: int,
    V: int,
    n: int,
    r: int,
    W_LM: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    iters: int,
    warmup: int,
) -> dict:
    cp = SharedTrunkCPSampler(H, n, r, V, W_LM, device, dtype)
    h = torch.randn(n, H, device=device, dtype=dtype)
    times = cuda_time_ms(lambda: cp.sample(h), iters=iters, warmup=warmup)
    del cp, h
    torch.cuda.empty_cache()
    return summary_stats(times)


def _apply_smoke(cfg: dict) -> None:
    cfg["block_sizes"] = [4, 16]
    cfg["ranks"] = [4, 16]
    cfg["vocabs"] = [32000, 152064]
    cfg["seq_lens"] = [1024]
    cfg["iters"] = 20
    cfg["warmup"] = 5
    cfg["run_name"] = cfg["run_name"] + "_smoke"


def _maybe_skip_config(n: int, r: int, V: int, H: int, device_gb: float = 120.0) -> bool:
    """Skip configs that would blow out HBM (soft cap at device_gb)."""
    log_p = _log_p_tensor_bytes(n, r, V)
    trunk = _pick_trunk_bytes_budget(V, H, torch.bfloat16)
    # Rough budget: trunk + log_p + workspace (~2x log_p).
    total = trunk + 3 * log_p
    return total > device_gb * 1e9


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.smoke:
        _apply_smoke(cfg)

    env_threads()
    set_seed(cfg.get("seed", 42))

    run_dir = make_run_dir(cfg["output"]["root"], cfg["run_name"])
    configure_logger(run_dir)
    dump_json(run_dir / "config.json", cfg)

    device = cfg["device"]
    dtype = dtype_from_str(cfg["dtype"])

    # ---- target forward pass -------------------------------------------------
    target_results: dict[int, dict] = {}
    if args.skip_target:
        H = int(cfg["hidden_size_override"])
        logger.warning(
            "Skipping target timing — using hidden_size_override="
            f"{H}. Ratios will be unavailable unless you inject numbers later."
        )
    else:
        target = load_target(cfg["target"], dtype, device)
        H = int(target.config.hidden_size)
        target_results = time_target(
            target,
            seq_lens=cfg["seq_lens"],
            iters=cfg["iters"],
            warmup=cfg["warmup"],
            device=device,
        )
        del target
        torch.cuda.empty_cache()

    # ---- CP + FF sampling grid ----------------------------------------------
    cp_rows: list[dict] = []
    ff_rows: list[dict] = []
    skipped: list[dict] = []

    for V in cfg["vocabs"]:
        logger.info(f"[V={V}] allocating shared trunk (V,H)=({V},{H}) ...")
        W_LM = torch.randn(V, H, device=device, dtype=dtype) * 0.02

        for n in cfg["block_sizes"]:
            ff_stat = _bench_ff(
                H, V, n, W_LM, device, dtype, cfg["iters"], cfg["warmup"]
            )
            ff_rows.append({"V": V, "n": n, **ff_stat})
            logger.info(
                f"  [V={V},n={n}] FF  median={ff_stat['median_ms']:.3f}ms"
            )

            for r in cfg["ranks"]:
                if _maybe_skip_config(n, r, V, H):
                    logger.warning(
                        f"  [V={V},n={n},r={r}] SKIP (estimated memory > budget)"
                    )
                    skipped.append({"V": V, "n": n, "r": r, "reason": "mem"})
                    continue
                cp_stat = _bench_cp(
                    H, V, n, r, W_LM, device, dtype, cfg["iters"], cfg["warmup"]
                )
                ff_median = ff_stat["median_ms"]
                overhead_x = cp_stat["median_ms"] / ff_median if ff_median else float("nan")
                cp_rows.append(
                    {
                        "V": V,
                        "n": n,
                        "r": r,
                        **cp_stat,
                        "overhead_vs_ff_x": overhead_x,
                    }
                )
                logger.info(
                    f"  [V={V},n={n},r={r}] CP  "
                    f"median={cp_stat['median_ms']:.3f}ms  "
                    f"({overhead_x:.2f}× FF)"
                )

        del W_LM
        torch.cuda.empty_cache()

    # ---- kill-criterion check -----------------------------------------------
    kill = cfg["kill_criterion"]
    cp_row = next(
        (
            r
            for r in cp_rows
            if r["V"] == kill["vocab_target"]
            and r["n"] == kill["block_size_target"]
            and r["r"] == kill["rank_target"]
        ),
        None,
    )
    tgt_stat = target_results.get(kill["seq_len_target"])
    if cp_row is not None and tgt_stat is not None:
        ratio = cp_row["median_ms"] / tgt_stat["median_ms"]
        decision = "KILL" if ratio > kill["ratio_threshold"] else "GO"
    else:
        ratio = None
        decision = "UNAVAILABLE"

    # Also report ratios across the full grid for each seq_len.
    ratio_grid: list[dict] = []
    for row in cp_rows:
        entry = {"V": row["V"], "n": row["n"], "r": row["r"], "cp_ms": row["median_ms"]}
        for L, s in target_results.items():
            entry[f"ratio_L{L}"] = row["median_ms"] / s["median_ms"]
        ratio_grid.append(entry)

    summary = {
        "hidden_size": H,
        "target": {str(L): s for L, s in target_results.items()},
        "ff": ff_rows,
        "cp": cp_rows,
        "ratio_grid": ratio_grid,
        "skipped": skipped,
        "kill_criterion": {
            "config": kill,
            "cp_ms": cp_row["median_ms"] if cp_row else None,
            "target_ms": tgt_stat["median_ms"] if tgt_stat else None,
            "ratio": ratio,
            "decision": decision,
        },
    }
    dump_json(run_dir / "summary.json", summary)
    logger.info(json.dumps(summary["kill_criterion"], indent=2))

    # ---- plots ---------------------------------------------------------------
    _plot_cp_vs_rank(cp_rows, run_dir)
    if target_results:
        _plot_ratio_grid(cp_rows, target_results, run_dir, kill["ratio_threshold"])

    print("\n" + "=" * 64)
    if ratio is not None:
        print(
            f"DECISION: {decision}   "
            f"(CP {cp_row['median_ms']:.2f}ms / target {tgt_stat['median_ms']:.2f}ms "
            f"= {ratio:.3f}  @ V={kill['vocab_target']},n={kill['block_size_target']},"
            f"r={kill['rank_target']},L={kill['seq_len_target']};  thr={kill['ratio_threshold']})"
        )
    else:
        print(f"DECISION: {decision}  (target timings unavailable)")
    print(f"Outputs: {run_dir}")
    print("=" * 64)


def _plot_cp_vs_rank(cp_rows: list[dict], run_dir: Path) -> None:
    vocabs = sorted({row["V"] for row in cp_rows})
    if not vocabs:
        return
    fig, axes = plt.subplots(
        1, len(vocabs), figsize=(5.2 * len(vocabs), 4.2), squeeze=False
    )
    for ax, V in zip(axes[0], vocabs):
        ns = sorted({row["n"] for row in cp_rows if row["V"] == V})
        for n in ns:
            xs, ys = [], []
            for r in sorted({row["r"] for row in cp_rows if row["V"] == V and row["n"] == n}):
                row = next(
                    row
                    for row in cp_rows
                    if row["V"] == V and row["n"] == n and row["r"] == r
                )
                xs.append(r)
                ys.append(row["median_ms"])
            ax.plot(xs, ys, marker="o", label=f"n={n}")
        ax.set_xlabel("CP rank")
        ax.set_ylabel("median CP sampling time (ms)")
        ax.set_title(f"V={V}")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("CP sampling latency vs rank")
    fig.tight_layout()
    fig.savefig(run_dir / "cp_latency.png", dpi=150)
    plt.close(fig)


def _plot_ratio_grid(
    cp_rows: list[dict],
    target_results: dict[int, dict],
    run_dir: Path,
    threshold: float,
) -> None:
    """One subplot per seq_len, showing t_CP / t_target for each (n, r) at each V."""
    seq_lens = sorted(target_results.keys())
    vocabs = sorted({row["V"] for row in cp_rows})
    if not seq_lens or not vocabs:
        return
    fig, axes = plt.subplots(
        len(seq_lens),
        len(vocabs),
        figsize=(5.0 * len(vocabs), 3.6 * len(seq_lens)),
        squeeze=False,
    )
    for i, L in enumerate(seq_lens):
        t_target = target_results[L]["median_ms"]
        for j, V in enumerate(vocabs):
            ax = axes[i][j]
            ns = sorted({row["n"] for row in cp_rows if row["V"] == V})
            for n in ns:
                rs = sorted({row["r"] for row in cp_rows if row["V"] == V and row["n"] == n})
                ys = [
                    next(
                        row["median_ms"]
                        for row in cp_rows
                        if row["V"] == V and row["n"] == n and row["r"] == r_
                    )
                    / t_target
                    for r_ in rs
                ]
                ax.plot(rs, ys, marker="o", label=f"n={n}")
            ax.axhline(threshold, color="red", ls="--", lw=1, label=f"thr={threshold}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("CP rank")
            ax.set_ylabel("t_CP / t_target")
            ax.set_title(f"V={V}, target L={L} ({t_target:.1f}ms)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("CP sampling cost as fraction of target forward")
    fig.tight_layout()
    fig.savefig(run_dir / "ratio_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
