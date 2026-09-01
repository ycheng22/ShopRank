"""Benchmark CPU latency of bge-reranker-v2-m3 for reranking.

Measures p50/p95/p99 over repeated runs with synthetic (query, passage) pairs
matching realistic ESCI data lengths. Prints results to stdout for recording
in DECISIONS.md.

Usage:
    python scripts/bench_rerank.py --top-k 50 --iterations 50
"""

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cross-encoder reranker latency on CPU")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates to rerank per query")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-v2-m3", help="Model name")
    parser.add_argument("--cache-dir", type=str, default="D:/huggingface_cache/hub", help="HuggingFace cache directory")
    args = parser.parse_args()

    print(f"Loading model: {args.model} on CPU...")
    load_start = time.perf_counter()

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.model, device="cpu", cache_folder=args.cache_dir)
    load_elapsed = time.perf_counter() - load_start
    print(f"Model loaded in {load_elapsed:.2f}s")

    # Synthetic data matching realistic ESCI lengths
    # Typical ESCI query: 3-8 words; typical product text: 20-80 words
    queries = [
        "wireless bluetooth headphones noise cancelling",
        "laptop stand adjustable",
        "organic green tea bags",
        "usb c hub multiport adapter",
        "running shoes men waterproof",
    ]

    passages = [
        f"Product {i}: This is a synthetic product description that simulates "
        f"a real Amazon product listing with title and description text combined. "
        f"It includes various product attributes, features, specifications, and "
        f"marketing copy that would typically appear in an e-commerce listing. "
        f"The text is approximately 50-80 words long to match real ESCI data. "
        f"Additional keywords: brand name, model number, color variant {i}."
        for i in range(args.top_k)
    ]

    latencies = []
    for iteration in range(args.iterations):
        query = queries[iteration % len(queries)]
        pairs = [(query, passage) for passage in passages]

        start = time.perf_counter()
        _scores = model.predict(pairs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        if iteration < 3 or iteration == args.iterations - 1:
            print(f"  Iteration {iteration + 1}/{args.iterations}: {elapsed_ms:.1f} ms")

    latencies_arr = np.array(latencies)

    # Skip first 3 iterations (warmup)
    if len(latencies_arr) > 5:
        warmup_latencies = latencies_arr[:3]
        latencies_arr = latencies_arr[3:]
        print(f"\nWarmup (first 3): mean={np.mean(warmup_latencies):.1f} ms")

    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean = float(np.mean(latencies_arr))

    print(f"\n{'='*60}")
    print("Rerank Benchmark Results")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print("Device:     CPU")
    print(f"Top-K:      {args.top_k}")
    print(f"Iterations: {args.iterations} (excluding 3 warmup)")
    print(f"{'='*60}")
    print(f"Mean:       {mean:.1f} ms")
    print(f"P50:        {p50:.1f} ms")
    print(f"P95:        {p95:.1f} ms")
    print(f"P99:        {p99:.1f} ms")
    print(f"{'='*60}")

    # Check against budget
    existing_p95 = 279.0  # Current hybrid pipeline p95
    total_p95 = existing_p95 + p95
    budget = 800.0
    headroom = budget - total_p95

    print("\nBudget analysis:")
    print(f"  Existing pipeline p95:  {existing_p95:.0f} ms")
    print(f"  Reranker p95:           {p95:.1f} ms")
    print(f"  Total estimated p95:    {total_p95:.1f} ms")
    print(f"  Budget:                 {budget:.0f} ms")
    print(f"  Headroom:               {headroom:.1f} ms")
    print(f"  Within budget:          {'YES' if headroom > 0 else 'NO'}")

    if headroom <= 0:
        print("\n  WARNING: Exceeds budget! Try --top-k 30")


if __name__ == "__main__":
    main()
