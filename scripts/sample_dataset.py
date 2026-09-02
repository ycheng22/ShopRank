import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SEED = 42


def compute_query_features(df):
    """
    Computes the 5 stratification labels for each query.
    Returns a dataframe of queries with their features.
    """
    # Group by query_id to get text
    query_texts = df[["query_id", "query"]].drop_duplicates(subset=["query_id"])

    # len_bin
    query_texts["tokens"] = query_texts["query"].apply(lambda x: len(str(x).split()))
    query_texts["len_bin"] = pd.cut(
        query_texts["tokens"],
        bins=[0, 2, 5, float("inf")],
        labels=["short", "medium", "long"],
        right=True,
    ).astype(str)

    # difficulty_bin: "hard" iff the query's fraction of E labels <= 0.312
    e_counts = df[df["esci_label"] == "E"].groupby("query_id").size()
    total_counts = df.groupby("query_id").size()
    e_frac = (
        e_counts.reindex(total_counts.index).fillna(0) / total_counts
    ).reset_index(name="e_fraction")
    e_frac["difficulty_bin"] = np.where(e_frac["e_fraction"] <= 0.312, "hard", "easy")

    # has_complement: True iff the query has at least one C label
    c_counts = df[df["esci_label"] == "C"].groupby("query_id").size()
    has_c = c_counts.reindex(total_counts.index).fillna(0) > 0
    has_c = has_c.reset_index(name="has_complement")

    # candidate_size_bin: "large" iff the query has >= 16 candidates
    cand_size = total_counts.reset_index(name="candidate_size")
    cand_size["candidate_size_bin"] = np.where(
        cand_size["candidate_size"] >= 16, "large", "small"
    )

    # Merge all
    q_df = (
        query_texts[["query_id", "query"]]
        .merge(query_texts[["query_id", "len_bin"]], on="query_id")
        .merge(e_frac[["query_id", "difficulty_bin"]], on="query_id")
        .merge(has_c, on="query_id")
        .merge(cand_size[["query_id", "candidate_size_bin"]], on="query_id")
    )

    # locale is always us for the main experimental set
    q_df["locale"] = "us"

    # rename query to text
    q_df = q_df.rename(columns={"query": "text"})

    return q_df


def print_distribution_report(df, split_name, total_queries=None):
    if total_queries is None:
        total_queries = len(df)

    logger.info(
        f"--- Distribution Report for {split_name} ({total_queries} queries) ---"
    )

    def report_dist(col, target_ratios):
        counts = df[col].value_counts(normalize=True)
        max_dev = 0
        for val, target in target_ratios.items():
            actual = counts.get(val, 0.0)
            dev = abs(actual - target)
            max_dev = max(max_dev, dev)
            logger.info(
                f"  {col}={val}: {actual:.1%} (target {target:.1%}, dev {dev:.1%})"
            )
        return max_dev

    dev_len = report_dist("len_bin", {"short": 0.25, "medium": 0.45, "long": 0.30})
    dev_diff = report_dist("difficulty_bin", {"easy": 0.40, "hard": 0.60})
    dev_cand = report_dist("candidate_size_bin", {"small": 0.40, "large": 0.60})

    c_ratio = df["has_complement"].mean()
    logger.info(f"  has_complement=True: {c_ratio:.1%} (target >= 20.0%)")

    # Check max deviation against equalities
    max_dev = max(dev_len, dev_diff, dev_cand)
    if c_ratio < 0.20:
        c_dev = 0.20 - c_ratio
        logger.info(f"  has_complement deviation: {c_dev:.1%}")
        max_dev = max(max_dev, c_dev)

    return max_dev


def get_sample(q_df, n=1500):
    rng = np.random.default_rng(SEED)

    target_len = {"short": 0.25, "medium": 0.45, "long": 0.30}
    target_diff = {"easy": 0.40, "hard": 0.60}
    target_cand = {"small": 0.40, "large": 0.60}

    pool_len = q_df["len_bin"].value_counts(normalize=True)
    pool_diff = q_df["difficulty_bin"].value_counts(normalize=True)
    pool_cand = q_df["candidate_size_bin"].value_counts(normalize=True)

    w = np.ones(len(q_df))
    w *= (
        q_df["len_bin"]
        .map(lambda x: target_len.get(x, 0) / (pool_len.get(x, 1e-5)))
        .values
    )
    w *= (
        q_df["difficulty_bin"]
        .map(lambda x: target_diff.get(x, 0) / (pool_diff.get(x, 1e-5)))
        .values
    )
    w *= (
        q_df["candidate_size_bin"]
        .map(lambda x: target_cand.get(x, 0) / (pool_cand.get(x, 1e-5)))
        .values
    )
    w *= q_df["has_complement"].map(lambda x: 1.5 if x else 0.8).values

    w = w / w.sum()

    best_sample_idx = None
    best_loss = float("inf")

    for _ in range(5000):
        sample_idx = rng.choice(q_df.index, size=n, replace=False, p=w)
        sample = q_df.loc[sample_idx]

        s_len = sample["len_bin"].value_counts(normalize=True)
        s_diff = sample["difficulty_bin"].value_counts(normalize=True)
        s_cand = sample["candidate_size_bin"].value_counts(normalize=True)
        s_comp = sample["has_complement"].mean()

        loss = 0
        loss += abs(s_len.get("short", 0) - 0.25)
        loss += abs(s_len.get("medium", 0) - 0.45)
        loss += abs(s_len.get("long", 0) - 0.30)
        loss += abs(s_diff.get("easy", 0) - 0.40)
        loss += abs(s_diff.get("hard", 0) - 0.60)
        loss += abs(s_cand.get("small", 0) - 0.40)
        loss += abs(s_cand.get("large", 0) - 0.60)

        if s_comp < 0.20:
            loss += (0.20 - s_comp) * 2

        if loss < best_loss:
            best_loss = loss
            best_sample_idx = sample_idx
            if loss < 0.05:
                break

    return q_df.loc[best_sample_idx].copy()


def split_stratified_exact(df, seed=42):
    """
    Splits the dataframe into train (60%), dev (20%), test (20%)
    exactly, stratified by the combination of dimensions.
    """
    df["stratum"] = (
        df["len_bin"]
        + "_"
        + df["difficulty_bin"]
        + "_"
        + df["candidate_size_bin"]
        + "_"
        + df["has_complement"].astype(str)
    )

    # Shuffle completely first
    df = df.sample(frac=1, random_state=seed)

    # Sort by stratum to group similar items together
    # Due to the stable nature of Python's sort, it keeps the random order within strata
    df = df.sort_values("stratum")

    # Assign cyclically to guarantee exactly 60/20/20 overall and perfectly balanced within each stratum
    # 3 train, 1 dev, 1 test = 5 items (60%, 20%, 20%)
    pattern = ["train", "train", "train", "dev", "test"]
    df["split"] = [pattern[i % 5] for i in range(len(df))]

    train_df = df[df["split"] == "train"].drop(columns=["stratum", "split"]).copy()
    dev_df = df[df["split"] == "dev"].drop(columns=["stratum", "split"]).copy()
    test_df = df[df["split"] == "test"].drop(columns=["stratum", "split"]).copy()

    return train_df, dev_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Sample ESCI dataset for ShopRank")
    parser.add_argument(
        "--examples",
        required=True,
        help="Path to shopping_queries_dataset_examples.parquet",
    )
    parser.add_argument(
        "--products",
        required=True,
        help="Path to shopping_queries_dataset_products.parquet",
    )
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument(
        "--max-products",
        type=int,
        default=30000,
        help="Maximum number of unique products allowed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, just print distribution",
    )
    args = parser.parse_args()

    logger.info(f"Using hardcoded seed: {SEED}")

    logger.info("Loading examples...")
    df_ex = pd.read_parquet(args.examples)
    logger.info("Loading products...")
    df_pr = pd.read_parquet(args.products)

    # Filter 1: small_version == 1 AND product_locale == 'us'
    initial_len = len(df_ex)
    df_ex = df_ex[(df_ex["small_version"] == 1) & (df_ex["product_locale"] == "us")]
    logger.info(
        f"Filtered examples from {initial_len} to {len(df_ex)} (small_version=1 & locale=us)"
    )

    logger.info("Computing query features...")
    q_df = compute_query_features(df_ex)

    logger.info("Sampling 1,500 queries to hit target ratios...")
    sampled_q = get_sample(q_df, n=1500)

    logger.info("Splitting 60/20/20 stratified by query...")
    train_q, dev_q, test_q = split_stratified_exact(sampled_q, seed=SEED)

    # Evaluate deviation on each split
    max_dev_train = print_distribution_report(train_q, "train (60%)")
    max_dev_dev = print_distribution_report(dev_q, "dev (20%)")
    max_dev_test = print_distribution_report(test_q, "test (20%)")

    # Calculate overall max deviation
    max_dev = max(max_dev_train, max_dev_dev, max_dev_test)

    # Collect candidate products for those queries
    query_ids = sampled_q["query_id"].unique()
    qrels = df_ex[df_ex["query_id"].isin(query_ids)][
        ["query_id", "product_id", "esci_label"]
    ].copy()

    # Check for empty qrels
    # A query has empty qrels if it is in the sampled queries but not in qrels
    empty_qrels_count = len(query_ids) - qrels["query_id"].nunique()
    logger.info(f"Queries with empty qrels: {empty_qrels_count}")

    unique_product_ids = qrels["product_id"].unique()
    logger.info(f"Total unique products needed: {len(unique_product_ids)}")

    if len(unique_product_ids) > args.max_products:
        logger.error(
            f"Number of unique products ({len(unique_product_ids)}) exceeds --max-products ({args.max_products})."
        )
        logger.error(
            "Stopping instead of dropping queries, as dropping would distort strata."
        )
        sys.exit(1)

    if max_dev > 0.02:
        logger.error(
            f"Maximum distribution deviation ({max_dev:.1%}) exceeds 2 percentage points!"
        )
        sys.exit(1)

    if args.dry_run:
        logger.info("--dry-run specified. Exiting without writing files.")
        sys.exit(0)

    # Build products df
    products = df_pr[df_pr["product_id"].isin(unique_product_ids)].copy()
    products["product_title"] = products["product_title"].fillna("")
    products["product_description"] = products["product_description"].fillna("")
    products["product_text"] = (
        products["product_title"] + " " + products["product_description"]
    )
    products["product_text"] = products["product_text"].str.strip()

    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    columns_splits = [
        "query_id",
        "text",
        "locale",
        "len_bin",
        "difficulty_bin",
        "has_complement",
        "candidate_size_bin",
    ]
    train_q[columns_splits].to_parquet(splits_dir / "train.parquet", index=False)
    dev_q[columns_splits].to_parquet(splits_dir / "dev.parquet", index=False)
    test_q[columns_splits].to_parquet(splits_dir / "test.parquet", index=False)

    # Write products
    products = products.drop_duplicates(subset=["product_id"])
    products[["product_id", "product_text"]].to_parquet(
        out_dir / "products.parquet", index=False
    )

    # Write qrels
    qrels.to_parquet(out_dir / "qrels.parquet", index=False)

    logger.info(f"Successfully wrote data to {out_dir}")


if __name__ == "__main__":
    main()
