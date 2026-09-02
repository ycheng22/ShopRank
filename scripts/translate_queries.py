"""Translate dev/test query strings to Chinese and French via provider abstraction.

Caches each translation on disk keyed by (model, target_lang, sha256(source)).
Re-running makes zero provider calls if the cache is warm.

Usage:
    python scripts/translate_queries.py --split dev --target-langs zh,fr --provider deepseek
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import asyncpg
import pandas as pd

from app.settings import get_settings
from providers.translation import translate_text

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _cache_key(model: str, target_lang: str, source_text: str) -> str:
    """Deterministic cache key: (model, target_lang, sha256(source))."""
    text_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    return f"{model}_{target_lang}_{text_hash}"


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _has_cjk(text: str) -> bool:
    """Check if text contains CJK characters."""
    for ch in text:
        if unicodedata.category(ch).startswith("Lo"):
            name = unicodedata.name(ch, "")
            if "CJK" in name or "CHINESE" in name or "HANGUL" in name:
                return True
    return False


def _has_latin(text: str) -> bool:
    """Check if text contains Latin script letters."""
    return bool(re.search(r"[a-zA-ZÀ-ÿ]", text))


def _sanity_check(source: str, translated: str, target_lang: str) -> str | None:
    """Return a rejection reason, or None if the translation passes.

    Checks:
    - Empty translation
    - Identical to source (failed to translate)
    - Wrong script (zh should have CJK, fr should have Latin)
    """
    if not translated or not translated.strip():
        return "empty"

    if translated.strip() == source.strip():
        return "identical_to_source"

    if target_lang == "zh" and not _has_cjk(translated):
        return "wrong_script_no_cjk"

    if target_lang == "fr" and not _has_latin(translated):
        return "wrong_script_no_latin"

    return None


async def translate_queries(
    split: str,
    target_langs: list[str],
    provider_name: str,
    cache_dir: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Translate queries and return a DataFrame of translations."""
    settings = get_settings()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load queries from DB
    pool = await asyncpg.create_pool(settings.database_url)
    try:
        rows = await pool.fetch(
            "SELECT query_id, text, locale FROM queries WHERE split = $1 AND locale = 'us' ORDER BY query_id",
            split,
        )
    finally:
        await pool.close()

    logger.info("Loaded %d queries from split '%s'", len(rows), split)

    # Determine model name for cache key
    model_name = f"{provider_name}_chat"

    records: list[dict[str, str]] = []
    stats = {
        "total": 0,
        "cached": 0,
        "translated": 0,
        "dropped": 0,
        "drop_reasons": {},
    }

    sem = asyncio.Semaphore(20)

    async def _process_one(lang: str, row: dict[str, str]) -> dict[str, str] | None:
        query_id = row["query_id"]
        source_text = row["text"]

        key = _cache_key(model_name, lang, source_text)
        cp = _cache_path(cache_dir, key)

        # Check cache
        if cp.exists():
            cached = json.loads(cp.read_text(encoding="utf-8"))
            translated = cached["translation"]
            return {
                "status": "cached",
                "query_id": f"{query_id}_{lang}",
                "locale": lang,
                "query_text": translated,
                "source_query_id": query_id,
                "source_text": source_text,
                "translated": translated,
            }

        if dry_run:
            logger.info("[DRY RUN] Would translate: %s → %s", source_text[:50], lang)
            return None

        async with sem:
            try:
                translated = await translate_text(
                    source_text,
                    lang,
                    provider_name,  # type: ignore[arg-type]
                    deepseek_api_key=settings.deepseek_api_key,
                    deepseek_base_url=settings.deepseek_base_url,
                    qwen_api_key=settings.qwen_api_key,
                    qwen_base_url=settings.qwen_base_url,
                    fallback_provider="alibaba"
                    if provider_name == "deepseek"
                    else "deepseek",
                    fallback_qwen_api_key=settings.qwen_api_key,
                    fallback_qwen_base_url=settings.qwen_base_url,
                    fallback_deepseek_api_key=settings.deepseek_api_key,
                    fallback_deepseek_base_url=settings.deepseek_base_url,
                )
            except Exception:
                logger.exception("Translation failed for query %s → %s", query_id, lang)
                return {"status": "error", "reason": "api_error", "lang": lang}

        # Cache to disk
        cp.write_text(
            json.dumps(
                {"source": source_text, "translation": translated, "lang": lang}
            ),
            encoding="utf-8",
        )

        return {
            "status": "translated",
            "query_id": f"{query_id}_{lang}",
            "locale": lang,
            "query_text": translated,
            "source_query_id": query_id,
            "source_text": source_text,
            "translated": translated,
        }

    tasks = []
    for lang in target_langs:
        for row in rows:
            stats["total"] += 1
            tasks.append(_process_one(lang, dict(row)))

    results = await asyncio.gather(*tasks)

    for res in results:
        if res is None:
            continue

        if res["status"] == "error":
            stats["dropped"] += 1
            reason = res["reason"]
            stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
            continue

        if res["status"] == "cached":
            stats["cached"] += 1
        elif res["status"] == "translated":
            stats["translated"] += 1

        # Sanity check
        reason = _sanity_check(res["source_text"], res["translated"], res["locale"])
        if reason is not None:
            stats["dropped"] += 1
            stats["drop_reasons"][reason] = stats["drop_reasons"].get(reason, 0) + 1
            logger.warning(
                "Dropped %s→%s: %s (source=%s, trans=%s)",
                res["source_query_id"],
                res["locale"],
                reason,
                res["source_text"][:40],
                res["translated"][:40],
            )
            continue

        records.append(
            {
                "query_id": res["query_id"],
                "locale": res["locale"],
                "query_text": res["query_text"],
                "source_query_id": res["source_query_id"],
            }
        )

    # Report
    logger.info("=" * 60)
    logger.info("Translation Summary")
    logger.info("=" * 60)
    logger.info("Total queries × langs:  %d", stats["total"])
    logger.info("From cache:             %d", stats["cached"])
    logger.info("Newly translated:       %d", stats["translated"])
    logger.info("Dropped:                %d", stats["dropped"])
    if stats["drop_reasons"]:
        for reason, count in stats["drop_reasons"].items():
            logger.info("  - %s: %d", reason, count)
    logger.info("Valid translations:     %d", len(records))
    logger.info("Provider calls made:    %d", stats["translated"])

    df = pd.DataFrame(records)
    return df


async def main_async(args: argparse.Namespace) -> None:
    target_langs = [lang.strip() for lang in args.target_langs.split(",")]
    cache_dir = Path(args.cache_dir)

    df = await translate_queries(
        split=args.split,
        target_langs=target_langs,
        provider_name=args.provider,
        cache_dir=cache_dir,
        dry_run=args.dry_run,
    )

    if df.empty:
        logger.warning("No translations produced.")
        return

    # Save parquet
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d translations to %s", len(df), out_path)

    # Insert into database
    if not args.dry_run:
        settings = get_settings()
        pool = await asyncpg.create_pool(settings.database_url)
        try:
            # Insert translated queries with ON CONFLICT to be idempotent
            for _, row in df.iterrows():
                await pool.execute(
                    """
                    INSERT INTO queries (query_id, text, locale, split)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (query_id) DO UPDATE SET text = EXCLUDED.text, locale = EXCLUDED.locale
                    """,
                    row["query_id"],
                    row["query_text"],
                    row["locale"],
                    args.split,
                )
            logger.info("Inserted %d translated queries into database", len(df))
        finally:
            await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate query strings for cross-lingual retrieval"
    )
    parser.add_argument(
        "--split", default="dev", help="Dataset split to translate (default: dev)"
    )
    parser.add_argument(
        "--target-langs", default="zh,fr", help="Comma-separated target languages"
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "alibaba"],
        help="Translation provider",
    )
    parser.add_argument(
        "--cache-dir", default=".cache/translations", help="Disk cache directory"
    )
    parser.add_argument(
        "--output",
        default="data/queries_multilingual.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be translated without calling APIs",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
