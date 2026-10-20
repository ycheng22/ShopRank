def test_app_imports() -> None:
    import retrieval_core

    assert retrieval_core.__version__ == "0.1.0"

    from app.main import app

    assert app.title == "ShopRank"


def test_get_config_hash_normalization() -> None:
    from app.routes.search import get_config_hash
    from core.pipeline import ShopRankPipelineConfig

    # BM25-only queries have no fusion; regardless of what fusion_method was passed,
    # it must normalize to 'none' and produce the identical hash.
    c_bm25_none = ShopRankPipelineConfig(
        use_bm25=True, use_dense=False, use_rerank=False, fusion_method="none", embed_dim=768
    )
    c_bm25_rrf = ShopRankPipelineConfig(
        use_bm25=True, use_dense=False, use_rerank=False, fusion_method="rrf", embed_dim=768
    )
    c_bm25_weighted = ShopRankPipelineConfig(
        use_bm25=True, use_dense=False, use_rerank=False, fusion_method="weighted", embed_dim=768
    )

    h_none = get_config_hash(c_bm25_none)
    assert h_none == get_config_hash(c_bm25_rrf)
    assert h_none == get_config_hash(c_bm25_weighted)

    # When use_dense is True and fusion_method is 'none', it normalizes to 'rrf'.
    c_dense_none = ShopRankPipelineConfig(
        use_bm25=True, use_dense=True, use_rerank=False, fusion_method="none", embed_dim=768
    )
    c_dense_rrf = ShopRankPipelineConfig(
        use_bm25=True, use_dense=True, use_rerank=False, fusion_method="rrf", embed_dim=768
    )
    assert get_config_hash(c_dense_none) == get_config_hash(c_dense_rrf)

