"""Estimate training runtime for a 2xT4 setup from a small throughput benchmark.

This is intentionally lightweight: it measures or accepts a single-GPU token
throughput, then scales it to a 2-GPU run with a configurable efficiency factor.
"""

from __future__ import annotations

import argparse
import math
import time


def estimate_training_tokens(num_examples: int, avg_tokens_per_example: int, epochs: float) -> int:
    return int(num_examples * avg_tokens_per_example * epochs)


def estimate_runtime_hours(total_tokens: int, tokens_per_sec: float) -> float:
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be positive")
    return total_tokens / tokens_per_sec / 3600.0


def estimate_multi_gpu_throughput(single_gpu_tokens_per_sec: float, num_gpus: int = 2, efficiency: float = 0.85) -> float:
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1")
    if not (0 < efficiency <= 1):
        raise ValueError("efficiency must be in the range (0, 1]")
    return single_gpu_tokens_per_sec * num_gpus * efficiency


def micro_benchmark_placeholder(batch_tokens: int, steps: int = 20) -> float:
    """Tiny deterministic stand-in for a real GPU benchmark.

    The script stays runnable on any machine. Replace this with a measured
    throughput from a real T4 run when you have one.
    """

    start = time.perf_counter()
    acc = 0
    for i in range(steps):
        acc += (batch_tokens * (i + 1)) % 97
    elapsed = time.perf_counter() - start
    _ = acc
    return batch_tokens * steps / max(elapsed, 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GPT-2 fine-tuning runtime on 2xT4.")
    parser.add_argument("--examples", type=int, default=50000, help="Number of training examples")
    parser.add_argument("--avg-tokens", type=int, default=180, help="Average tokens per example")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of epochs")
    parser.add_argument(
        "--single-gpu-tokens-per-sec",
        type=float,
        default=None,
        help="Measured throughput from one T4 in tokens/sec. If omitted, a local placeholder benchmark is used.",
    )
    parser.add_argument("--num-gpus", type=int, default=2, help="GPU count to estimate for")
    parser.add_argument("--efficiency", type=float, default=0.85, help="Multi-GPU scaling efficiency")
    parser.add_argument("--batch-tokens", type=int, default=4096, help="Placeholder benchmark batch size in tokens")
    args = parser.parse_args()

    single_gpu_tps = args.single_gpu_tokens_per_sec
    if single_gpu_tps is None:
        single_gpu_tps = micro_benchmark_placeholder(args.batch_tokens)

    multi_gpu_tps = estimate_multi_gpu_throughput(single_gpu_tps, num_gpus=args.num_gpus, efficiency=args.efficiency)
    total_tokens = estimate_training_tokens(args.examples, args.avg_tokens, args.epochs)

    single_gpu_hours = estimate_runtime_hours(total_tokens, single_gpu_tps)
    multi_gpu_hours = estimate_runtime_hours(total_tokens, multi_gpu_tps)
    speedup = single_gpu_hours / multi_gpu_hours if multi_gpu_hours > 0 else math.inf

    print("Training estimate")
    print(f"  examples: {args.examples}")
    print(f"  avg tokens/example: {args.avg_tokens}")
    print(f"  epochs: {args.epochs}")
    print(f"  total tokens: {total_tokens}")
    print(f"  single-GPU throughput: {single_gpu_tps:.2f} tokens/sec")
    print(f"  estimated {args.num_gpus}xT4 throughput (@{args.efficiency:.0%} efficiency): {multi_gpu_tps:.2f} tokens/sec")
    print(f"  estimated 1xT4 runtime: {single_gpu_hours:.2f} hours")
    print(f"  estimated {args.num_gpus}xT4 runtime: {multi_gpu_hours:.2f} hours")
    print(f"  speedup vs 1xT4: {speedup:.2f}x")


if __name__ == "__main__":
    main()