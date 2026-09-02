from typing import Literal

from retrieval_core.models import ScoredHit


def fuse(
    runs: dict[str, list[ScoredHit]],
    method: Literal["rrf", "weighted", "none"],
    params: dict | None = None,
) -> list[ScoredHit]:
    if params is None:
        params = {}

    # Aggregate hits by product_id to preserve all raw scores
    aggregated = {}
    for retriever_name, hits in runs.items():
        for hit in hits:
            if hit.product_id not in aggregated:
                aggregated[hit.product_id] = {
                    "product_id": hit.product_id,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "hits_by_retriever": {},
                }

            # Map score to breakdown
            if retriever_name == "bm25":
                aggregated[hit.product_id]["bm25_score"] = hit.raw_score
            elif retriever_name == "dense":
                aggregated[hit.product_id]["dense_score"] = hit.raw_score

            aggregated[hit.product_id]["hits_by_retriever"][retriever_name] = hit

    if method == "none":
        # Union without fusion. Sort by the highest raw_score across all retrievers.
        # This will produce terrible ranking if scales mismatch, which serves as a baseline to prove fusion is needed.
        fused_results = []
        for pid, data in aggregated.items():
            max_raw = 0.0
            for name, hit in data["hits_by_retriever"].items():
                max_raw = max(max_raw, hit.raw_score)

            first_hit = next(iter(data["hits_by_retriever"].values())).model_copy(
                deep=True
            )
            first_hit.breakdown.bm25_score = data["bm25_score"]
            first_hit.breakdown.dense_score = data["dense_score"]
            first_hit.breakdown.fused_score = max_raw
            first_hit.raw_score = max_raw
            first_hit.retriever_name = "hybrid_none"
            fused_results.append(first_hit)

        fused_results.sort(key=lambda x: x.raw_score, reverse=True)
        # Update ranks
        for i, hit in enumerate(fused_results):
            hit.rank = i + 1
        return fused_results

    elif method == "rrf":
        k = params.get("k", 60)
        # Compute RRF score
        for pid, data in aggregated.items():
            rrf_score = 0.0
            for name, hit in data["hits_by_retriever"].items():
                rrf_score += 1.0 / (k + hit.rank)
            data["fused_score"] = rrf_score

        # Sort by fused_score
        sorted_pids = sorted(
            aggregated.keys(),
            key=lambda pid: aggregated[pid]["fused_score"],
            reverse=True,
        )

    elif method == "weighted":
        # Min-max normalize per run
        normalized_runs = {}
        for name, hits in runs.items():
            if not hits:
                continue
            scores = [h.raw_score for h in hits]
            min_s = min(scores)
            max_s = max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            normalized_runs[name] = {
                h.product_id: (h.raw_score - min_s) / range_s for h in hits
            }

        weights = params.get("weights", {"bm25": 0.5, "dense": 0.5})

        for pid, data in aggregated.items():
            weighted_score = 0.0
            for name in data["hits_by_retriever"]:
                norm_score = normalized_runs[name].get(pid, 0.0)
                w = weights.get(name, 0.5)
                weighted_score += norm_score * w
            data["fused_score"] = weighted_score

        sorted_pids = sorted(
            aggregated.keys(),
            key=lambda pid: aggregated[pid]["fused_score"],
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    # Build final list
    results = []
    for rank_0_indexed, pid in enumerate(sorted_pids):
        data = aggregated[pid]
        # Clone the first hit to base our result on
        first_hit = next(iter(data["hits_by_retriever"].values())).model_copy(deep=True)

        # We must create a new ScoredHit or modify the existing one safely
        first_hit.raw_score = data["fused_score"]
        first_hit.rank = rank_0_indexed + 1
        first_hit.retriever_name = "hybrid"

        # Populate ScoreBreakdown
        first_hit.breakdown.bm25_score = data["bm25_score"]
        first_hit.breakdown.dense_score = data["dense_score"]
        first_hit.breakdown.fused_score = data["fused_score"]

        results.append(first_hit)

    return results
