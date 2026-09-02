from unittest.mock import MagicMock

import pytest
from retrieval_core.models import Query, ScoreBreakdown, ScoredHit

from core.rerankers.cross_encoder import rerank


@pytest.fixture(autouse=True)
def _mock_cross_encoder_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda pairs: [
        float(len(pairs) - i) for i in range(len(pairs))
    ]
    monkeypatch.setattr("core.rerankers.cross_encoder._model", mock_model)


def _make_hit(
    product_id: str,
    *,
    bm25_score: float = 0.0,
    dense_score: float = 0.0,
    fused_score: float = 0.0,
    rank: int = 1,
) -> ScoredHit:
    return ScoredHit(
        product_id=product_id,
        raw_score=fused_score,
        retriever_name="hybrid",
        rank=rank,
        breakdown=ScoreBreakdown(
            bm25_score=bm25_score,
            dense_score=dense_score,
            fused_score=fused_score,
        ),
    )


class TestRerankerBreakdownPreservation:
    """Acceptance criterion 2: reranked results must carry ALL breakdown fields."""

    def test_rerank_preserves_prior_scores(self) -> None:
        """bm25_score, dense_score, fused_score must survive reranking."""
        hits = [
            _make_hit("A", bm25_score=8.0, dense_score=0.9, fused_score=0.5, rank=1),
            _make_hit("B", bm25_score=3.0, dense_score=0.7, fused_score=0.3, rank=2),
        ]
        product_texts = {
            "A": "Wireless bluetooth headphones with noise cancelling",
            "B": "USB-C charging cable 6ft for smartphones",
        }

        result = rerank(
            Query(text="headphones"),
            hits,
            top_k=2,
            product_texts=product_texts,
        )

        for hit in result:
            # Prior scores must be preserved (not zeroed out)
            assert hit.breakdown.bm25_score > 0
            assert hit.breakdown.dense_score > 0
            assert hit.breakdown.fused_score > 0
            # New fields must be set
            assert hit.breakdown.rerank_score != 0.0
            assert hit.breakdown.rank_before_rerank is not None

    def test_rerank_sets_rank_before_rerank(self) -> None:
        """rank_before_rerank reflects the original fusion rank."""
        hits = [
            _make_hit("A", fused_score=0.9, rank=1),
            _make_hit("B", fused_score=0.5, rank=2),
            _make_hit("C", fused_score=0.3, rank=3),
        ]
        product_texts = {
            "A": "Product A description text for scoring",
            "B": "Product B description text for scoring",
            "C": "Product C description text for scoring",
        }

        result = rerank(
            Query(text="test query"),
            hits,
            top_k=3,
            product_texts=product_texts,
        )

        original_ranks = {h.product_id: h.breakdown.rank_before_rerank for h in result}
        assert original_ranks["A"] == 1
        assert original_ranks["B"] == 2
        assert original_ranks["C"] == 3

    def test_rerank_empty_input(self) -> None:
        """Empty hit list returns empty."""
        result = rerank(Query(text="test"), [], top_k=10, product_texts={})
        assert result == []

    def test_rerank_missing_product_text(self) -> None:
        """Hits without product text are excluded from reranking."""
        hits = [
            _make_hit("A", fused_score=0.9, rank=1),
            _make_hit("B", fused_score=0.5, rank=2),
        ]
        # Only A has text
        product_texts = {"A": "Product A description"}

        result = rerank(
            Query(text="test"),
            hits,
            top_k=2,
            product_texts=product_texts,
        )

        # Only A should be in the result since B has no text
        assert len(result) == 1
        assert result[0].product_id == "A"
