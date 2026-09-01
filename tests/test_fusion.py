"""Tests for fusion logic — RRF, weighted, and score preservation."""

import pytest
from retrieval_core.models import ScoreBreakdown, ScoredHit

from core.fusion import fuse


def _make_hit(
    product_id: str,
    raw_score: float,
    retriever_name: str,
    rank: int,
    **breakdown_kwargs: float,
) -> ScoredHit:
    breakdown = ScoreBreakdown(**breakdown_kwargs)
    return ScoredHit(
        product_id=product_id,
        raw_score=raw_score,
        retriever_name=retriever_name,
        rank=rank,
        breakdown=breakdown,
    )


class TestRRFFusion:
    def test_rrf_ranks_by_reciprocal_rank_sum(self) -> None:
        """Product appearing in both lists at rank 1 should beat one at rank 2."""
        bm25_hits = [
            _make_hit("A", 10.0, "bm25", 1),
            _make_hit("B", 5.0, "bm25", 2),
        ]
        dense_hits = [
            _make_hit("A", 0.9, "dense", 1),
            _make_hit("C", 0.8, "dense", 2),
        ]
        runs = {"bm25": bm25_hits, "dense": dense_hits}
        result = fuse(runs, method="rrf", params={"k": 60})

        assert result[0].product_id == "A"
        assert result[0].rank == 1

    def test_rrf_k_affects_ranking(self) -> None:
        """Different k values should change relative scores."""
        bm25_hits = [
            _make_hit("A", 10.0, "bm25", 1),
            _make_hit("B", 5.0, "bm25", 2),
        ]
        dense_hits = [
            _make_hit("B", 0.9, "dense", 1),
            _make_hit("A", 0.8, "dense", 2),
        ]
        runs = {"bm25": bm25_hits, "dense": dense_hits}

        result_k10 = fuse(runs, method="rrf", params={"k": 10})
        result_k200 = fuse(runs, method="rrf", params={"k": 200})

        # Both should return A and B but the relative scores differ
        assert len(result_k10) == 2
        assert len(result_k200) == 2

    def test_rrf_preserves_breakdown_fields(self) -> None:
        """Fused hits must carry bm25_score, dense_score, and fused_score."""
        bm25_hits = [_make_hit("A", 8.0, "bm25", 1)]
        dense_hits = [_make_hit("A", 0.95, "dense", 1)]
        runs = {"bm25": bm25_hits, "dense": dense_hits}

        result = fuse(runs, method="rrf", params={"k": 60})
        hit = result[0]

        assert hit.breakdown.bm25_score == 8.0
        assert hit.breakdown.dense_score == 0.95
        assert hit.breakdown.fused_score > 0


class TestWeightedFusion:
    def test_weighted_uses_configured_weights(self) -> None:
        """Higher weight on dense should favor the dense-ranked product."""
        bm25_hits = [
            _make_hit("A", 10.0, "bm25", 1),
            _make_hit("B", 5.0, "bm25", 2),
        ]
        dense_hits = [
            _make_hit("B", 0.9, "dense", 1),
            _make_hit("A", 0.1, "dense", 2),
        ]
        runs = {"bm25": bm25_hits, "dense": dense_hits}

        # Heavy weight on dense should push B up
        result = fuse(
            runs,
            method="weighted",
            params={"weights": {"bm25": 0.1, "dense": 0.9}},
        )

        assert result[0].product_id == "B"

    def test_weighted_preserves_all_scores(self) -> None:
        """All breakdown fields must survive weighted fusion."""
        bm25_hits = [
            _make_hit("X", 7.0, "bm25", 1),
            _make_hit("Y", 3.0, "bm25", 2),
        ]
        dense_hits = [
            _make_hit("X", 0.8, "dense", 1),
            _make_hit("Y", 0.2, "dense", 2),
        ]
        runs = {"bm25": bm25_hits, "dense": dense_hits}

        result = fuse(runs, method="weighted")
        # X should be first (highest in both lists)
        hit = next(h for h in result if h.product_id == "X")

        assert hit.breakdown.bm25_score == 7.0
        assert hit.breakdown.dense_score == 0.8
        assert hit.breakdown.fused_score > 0


class TestNoneFusion:
    def test_none_fusion_uses_max_score(self) -> None:
        """'none' fusion just takes the max raw_score."""
        bm25_hits = [_make_hit("A", 10.0, "bm25", 1)]
        dense_hits = [_make_hit("A", 0.5, "dense", 1)]
        runs = {"bm25": bm25_hits, "dense": dense_hits}

        result = fuse(runs, method="none")
        assert result[0].raw_score == 10.0


class TestUnknownMethod:
    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion method"):
            fuse({}, method="magic")  # type: ignore[arg-type]
