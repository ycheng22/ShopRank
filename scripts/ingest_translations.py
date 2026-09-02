import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def main():
    engine = create_engine(os.environ["DATABASE_URL"])

    print("Loading translated queries...")
    parquet_path = Path("data/queries_multilingual.parquet")
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)

    # df has: query_id, locale, query_text, source_query_id
    # We need to map query_text -> text, and fetch the rest from DB

    # fetch original queries
    with engine.connect() as conn:
        original = pd.read_sql(
            "SELECT query_id as source_query_id, len_bin, difficulty_bin, has_complement, candidate_size_bin, split FROM queries WHERE locale = 'us'",
            conn,
        )

    # merge
    merged = df.merge(original, on="source_query_id", how="left")

    # rename query_text to text
    merged = merged.rename(columns={"query_text": "text"})

    # drop source_query_id
    merged = merged.drop(columns=["source_query_id"])

    # ensure no duplicates if re-running
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM queries WHERE locale IN ('zh', 'fr')"))

    # insert
    print(f"Inserting {len(merged)} translated queries into DB...")
    merged.to_sql(
        "queries",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print("Done!")


if __name__ == "__main__":
    main()
