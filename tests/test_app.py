def test_app_imports() -> None:
    import retrieval_core

    assert retrieval_core.__version__ == "0.1.0"

    from app.main import app

    assert app.title == "ShopRank"
