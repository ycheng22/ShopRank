import pytest

from evals.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg() -> None:
    # gains [3,0,2,1,0] -> NDCG@5 == 0.9304 (4 dp).
    qrels = {
        "doc1": "Exact",
        "doc2": "Irrelevant",
        "doc3": "Substitute",
        "doc4": "Complement",
        "doc5": "Irrelevant",
    }
    results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    score, skipped = ndcg_at_k(qrels, results, k=5)
    assert skipped == 0
    assert pytest.approx(score, abs=0.0001) == 0.9304


def test_ndcg_reversed() -> None:
    # The fully reversed ranking [0,1,2,0,3] scores strictly lower than case 1
    qrels = {
        "doc1": "Exact",
        "doc2": "Irrelevant",
        "doc3": "Substitute",
        "doc4": "Complement",
        "doc5": "Irrelevant",
    }
    results_fwd = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    results_rev = ["doc5", "doc4", "doc3", "doc2", "doc1"]
    score_fwd, _ = ndcg_at_k(qrels, results_fwd, k=5)
    score_rev, _ = ndcg_at_k(qrels, results_rev, k=5)
    assert score_rev < score_fwd


def test_mrr() -> None:
    # first relevant hit at rank 1 -> MRR == 1.0
    qrels_1 = {"doc1": "Substitute"}
    results_1 = ["doc1", "doc2", "doc3"]
    score_1, skipped_1 = mrr_at_k(qrels_1, results_1, k=5)
    assert skipped_1 == 0
    assert pytest.approx(score_1, abs=0.0001) == 1.0

    # first relevant hit at rank 3 -> MRR == 1/3
    qrels_3 = {"doc3": "Exact"}
    results_3 = ["doc1", "doc2", "doc3"]
    score_3, skipped_3 = mrr_at_k(qrels_3, results_3, k=5)
    assert skipped_3 == 0
    assert pytest.approx(score_3, abs=0.0001) == 1.0 / 3.0


def test_recall() -> None:
    # 3 of 4 relevant products retrieved -> recall == 0.75
    qrels = {
        "doc1": "Exact",
        "doc2": "Substitute",
        "doc3": "Complement",
        "doc4": "Exact",
    }
    results = ["doc1", "doc2", "doc3", "doc5"]
    score, skipped = recall_at_k(qrels, results, k=5)
    assert skipped == 0
    assert score == 0.75


def test_empty_qrels_skipped() -> None:
    # a query with empty qrels is skipped, and the reported skipped count is 1.
    qrels: dict[str, str] = {}
    results = ["doc1"]
    for metric in [ndcg_at_k, mrr_at_k, recall_at_k]:
        score, skipped = metric(qrels, results, k=5)
        assert skipped == 1
        assert score == 0.0


def test_out_of_bounds_k() -> None:
    # recall@50 on a 5-item result list does not raise
    qrels = {"doc1": "Exact"}
    results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    score, skipped = recall_at_k(qrels, results, k=50)
    assert skipped == 0
    assert score == 1.0
