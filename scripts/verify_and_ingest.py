import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def run_sample_dataset(out_dir):
    print(f"Running sample_dataset.py for {out_dir}...")
    cmd = [
        sys.executable,
        "scripts/sample_dataset.py",
        "--examples",
        "data/raw/shopping_queries_dataset_examples.parquet",
        "--products",
        "data/raw/shopping_queries_dataset_products.parquet",
        "--out-dir",
        out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to run script. Output:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    # Extract deviations from stderr/stdout
    output = result.stdout + result.stderr
    print("Script output snippet (deviations):")
    for line in output.split("\n"):
        if (
            "Maximum distribution deviation" in line
            or "dev " in line
            or "has_complement deviation" in line
        ):
            print(line)


def compare_dirs(dir1, dir2):
    print("\nComparing outputs for byte-level idempotency...")
    files_to_check = [
        "products.parquet",
        "qrels.parquet",
        "splits/train.parquet",
        "splits/dev.parquet",
        "splits/test.parquet",
    ]

    all_match = True
    for f in files_to_check:
        f1 = Path(dir1) / f
        f2 = Path(dir2) / f

        if not f1.exists() or not f2.exists():
            print(f"Missing file: {f}")
            all_match = False
            continue

        md5_1 = hashlib.md5(f1.read_bytes()).hexdigest()
        md5_2 = hashlib.md5(f2.read_bytes()).hexdigest()

        if md5_1 == md5_2:
            print(f"{f}: MATCH ({md5_1})")
        else:
            print(f"{f}: MISMATCH ({md5_1} vs {md5_2})")
            all_match = False

    if all_match:
        print("Idempotency check PASSED.")
    else:
        print("Idempotency check FAILED.")


def check_leakage(out_dir):
    print("\nChecking for candidate leakage across splits...")
    qrels = pd.read_parquet(Path(out_dir) / "qrels.parquet")
    train_q = pd.read_parquet(Path(out_dir) / "splits/train.parquet")[
        "query_id"
    ].tolist()
    dev_q = pd.read_parquet(Path(out_dir) / "splits/dev.parquet")["query_id"].tolist()
    test_q = pd.read_parquet(Path(out_dir) / "splits/test.parquet")["query_id"].tolist()

    # Get products for each split
    train_p = set(qrels[qrels["query_id"].isin(train_q)]["product_id"])
    dev_p = set(qrels[qrels["query_id"].isin(dev_q)]["product_id"])
    test_p = set(qrels[qrels["query_id"].isin(test_q)]["product_id"])

    print(f"Train candidates: {len(train_p)}")
    print(f"Dev candidates: {len(dev_p)}")
    print(f"Test candidates: {len(test_p)}")

    leak_train_dev = train_p.intersection(dev_p)
    leak_train_test = train_p.intersection(test_p)
    leak_dev_test = dev_p.intersection(test_p)

    total_leaks = len(leak_train_dev) + len(leak_train_test) + len(leak_dev_test)
    if total_leaks == 0:
        print("Leakage check PASSED. No candidates leaked across splits.")
    else:
        print(
            f"Leakage check FAILED. Overlaps: train-dev({len(leak_train_dev)}), train-test({len(leak_train_test)}), dev-test({len(leak_dev_test)})"
        )


def ingest_to_neon(out_dir):
    print("\nIngesting into Neon database...")
    from app.settings import get_settings

    settings = get_settings()
    db_url = settings.database_url
    if not db_url:
        print("DATABASE_URL is not set.")
        sys.exit(1)

    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    engine = create_engine(db_url)

    # Run schema.sql
    with engine.begin() as conn, open("scripts/schema.sql", "r") as f:
        from sqlalchemy import text

        # split statements
        sql = f.read()
        for stmt in sql.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
    print("Schema applied.")

    # Load data
    print("Loading data into DB (this might take a minute)...")
    products = pd.read_parquet(Path(out_dir) / "products.parquet")
    qrels = pd.read_parquet(Path(out_dir) / "qrels.parquet")

    train_q = pd.read_parquet(Path(out_dir) / "splits/train.parquet")
    train_q["split"] = "train"
    dev_q = pd.read_parquet(Path(out_dir) / "splits/dev.parquet")
    dev_q["split"] = "dev"
    test_q = pd.read_parquet(Path(out_dir) / "splits/test.parquet")
    test_q["split"] = "test"
    queries = pd.concat([train_q, dev_q, test_q], ignore_index=True)

    # to_sql
    try:
        print(f"Inserting {len(products)} products...")
        # Since products has an embedding column which is not filled yet, we just insert the available columns
        # To handle ON CONFLICT, we can just use to_sql with if_exists='append' (actually if they exist it will fail).
        # Better to just wipe tables for this test
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE qrels, queries, products CASCADE;"))

        products.to_sql(
            "products",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"Inserting {len(queries)} queries...")
        queries.to_sql(
            "queries",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        print(f"Inserting {len(qrels)} qrels...")
        qrels.to_sql(
            "qrels",
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
    except Exception as e:
        print(f"Ingest failed: {e}")
        return

    # Verify counts
    with engine.connect() as conn:
        p_count = conn.execute(text("SELECT count(*) FROM products")).scalar()
        q_count = conn.execute(text("SELECT count(*) FROM queries")).scalar()
        qr_count = conn.execute(text("SELECT count(*) FROM qrels")).scalar()

    print(f"DB counts: products={p_count}, queries={q_count}, qrels={qr_count}")
    if p_count == len(products) and q_count == len(queries) and qr_count == len(qrels):
        print("Ingest check PASSED.")
    else:
        print("Ingest check FAILED. Counts do not match.")


if __name__ == "__main__":
    run_sample_dataset("data/run1")
    run_sample_dataset("data/run2")
    compare_dirs("data/run1", "data/run2")
    check_leakage("data/run1")
    ingest_to_neon("data/run1")
