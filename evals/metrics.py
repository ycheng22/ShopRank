import math

GAIN_MAP = {
    "Exact": 3,
    "Substitute": 2,
    "Complement": 1,
    "Irrelevant": 0,
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0,
}


def _get_gain(label: str) -> int:
    return GAIN_MAP.get(label, 0)


def ndcg_at_k(qrels: dict[str, str], results: list[str], k: int) -> tuple[float, int]:
    """
    Computes NDCG@k.
    Returns (score, skipped_count).
    Empty qrels returns (0.0, 1).
    """
    if not qrels:
        return 0.0, 1

    dcg = 0.0
    for i, doc_id in enumerate(results[:k]):
        rank = i + 1
        if doc_id in qrels:
            gain = _get_gain(qrels[doc_id])
            if gain > 0:
                dcg += gain / math.log2(rank + 1)

    ideal_gains = sorted([_get_gain(label) for label in qrels.values()], reverse=True)
    idcg = 0.0
    for i, gain in enumerate(ideal_gains[:k]):
        if gain == 0:
            break
        rank = i + 1
        idcg += gain / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0, 0

    return dcg / idcg, 0


def recall_at_k(qrels: dict[str, str], results: list[str], k: int) -> tuple[float, int]:
    """
    Computes Recall@k.
    Returns (score, skipped_count).
    Empty qrels returns (0.0, 1).
    """
    if not qrels:
        return 0.0, 1

    relevant_docs = {doc_id for doc_id, label in qrels.items() if _get_gain(label) > 0}
    if not relevant_docs:
        return 0.0, 0

    retrieved_relevant = 0
    for doc_id in results[:k]:
        if doc_id in relevant_docs:
            retrieved_relevant += 1

    return retrieved_relevant / len(relevant_docs), 0


def mrr_at_k(qrels: dict[str, str], results: list[str], k: int) -> tuple[float, int]:
    """
    Computes MRR@k.
    Returns (score, skipped_count).
    Empty qrels returns (0.0, 1).
    """
    if not qrels:
        return 0.0, 1

    for i, doc_id in enumerate(results[:k]):
        if doc_id in qrels and _get_gain(qrels[doc_id]) > 0:
            return 1.0 / (i + 1), 0

    return 0.0, 0
