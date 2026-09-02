"""Tests for translation sanity checks and cache key generation."""

from scripts.translate_queries import _cache_key, _has_cjk, _has_latin, _sanity_check


class TestSanityCheck:
    def test_empty_translation_rejected(self) -> None:
        assert _sanity_check("hello", "", "zh") == "empty"
        assert _sanity_check("hello", "   ", "zh") == "empty"

    def test_identical_translation_rejected(self) -> None:
        assert (
            _sanity_check("hello world", "hello world", "zh") == "identical_to_source"
        )

    def test_wrong_script_zh_rejected(self) -> None:
        """Chinese translation without CJK characters is rejected."""
        assert _sanity_check("hello", "bonjour", "zh") == "wrong_script_no_cjk"

    def test_wrong_script_fr_rejected(self) -> None:
        """French translation without Latin characters is rejected."""
        assert _sanity_check("hello", "你好世界", "fr") == "wrong_script_no_latin"

    def test_valid_zh_passes(self) -> None:
        assert _sanity_check("headphones", "耳机", "zh") is None

    def test_valid_fr_passes(self) -> None:
        assert _sanity_check("headphones", "écouteurs", "fr") is None


class TestCJKDetection:
    def test_has_cjk_chinese(self) -> None:
        assert _has_cjk("无线蓝牙耳机")

    def test_has_cjk_mixed(self) -> None:
        assert _has_cjk("iPhone 15 保护壳")

    def test_no_cjk_latin(self) -> None:
        assert not _has_cjk("bluetooth headphones")


class TestLatinDetection:
    def test_has_latin_english(self) -> None:
        assert _has_latin("bluetooth headphones")

    def test_has_latin_french(self) -> None:
        assert _has_latin("écouteurs sans fil")

    def test_no_latin_cjk_only(self) -> None:
        assert not _has_latin("无线蓝牙耳机")


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("deepseek_chat", "zh", "hello")
        k2 = _cache_key("deepseek_chat", "zh", "hello")
        assert k1 == k2

    def test_different_lang_different_key(self) -> None:
        k1 = _cache_key("deepseek_chat", "zh", "hello")
        k2 = _cache_key("deepseek_chat", "fr", "hello")
        assert k1 != k2

    def test_different_model_different_key(self) -> None:
        k1 = _cache_key("deepseek_chat", "zh", "hello")
        k2 = _cache_key("alibaba_chat", "zh", "hello")
        assert k1 != k2

    def test_different_text_different_key(self) -> None:
        k1 = _cache_key("deepseek_chat", "zh", "hello")
        k2 = _cache_key("deepseek_chat", "zh", "world")
        assert k1 != k2
