# Failure Notes & Hard Negatives Analysis

This document provides a comprehensive analysis of the failure modes, misranking behaviors, and hard negatives discovered in the ShopRank retrieval pipeline on the Amazon ESCI benchmark.

---

## 1. Storage & Location of Raw Failure Data

| Artifact / File | Location | Content |
| --- | --- | --- |
| **Diagnostics Document** | [`docs/DIAGNOSTICS.md`](file:///d:/Github_Clones/ShopRank/docs/DIAGNOSTICS.md) | High-level diagnostic write-up of top S/C/I failure modes. |
| **Mined Hard Negatives** | [`data/hard_negatives.parquet`](file:///d:/Github_Clones/ShopRank/data/hard_negatives.parquet) | 292 extracted hard negatives from the 300 dev query set. |
| **Mining Script** | [`scripts/mine_hard_negatives.py`](file:///d:/Github_Clones/ShopRank/scripts/mine_hard_negatives.py) | Automated mining tool configured with S/C/I thresholds. |
| **README Summary** | [`README.md#6-failure-modes--hard-negatives`](file:///d:/Github_Clones/ShopRank/README.md) | Overview of primary failure modes and mitigations. |

---

## 2. Hard Negative Categories & Thresholds

Hard negatives were mined from the dev split using the hybrid pipeline without reranking, filtering for:
- **Substitute (S)**: Ranked at **Rank 1** (53 occurrences).
- **Complement (C)**: Ranked in **Top 5** (69 occurrences).
- **Irrelevant (I)**: Ranked in **Top 10** (170 occurrences).

Total mined hard negatives: **292**.

---

## 3. Top Egregious Failure Cases

### Case 1: Substitute (S) Misranking (Rank = 1)
- **Query ID / Text**: `the not a cat cat`
- **Top-Ranked Product**: `I'm Here Live I'm Not A Cat: I'm Here Live I'm Not A Cat Fun`
- **Gold Label**: `S` (Substitute / Parody)
- **Why it failed**: Exact lexical ngram match ("Not A Cat") strongly dominated both the BM25 score and dense token overlap, causing meme merchandise to outrank genuine cat products.

### Case 2: Complement (C) Misranking (Rank = 1)
- **Query ID / Text**: `under a white sky: the nature of the future`
- **Top-Ranked Product**: `Summary and Analysis of Under a White Sky: The Nature of the`
- **Gold Label**: `C` (Complement / Companion Guide)
- **Why it failed**: The study guide shares 95% of the title tokens with Elizabeth Kolbert's original book. Both dense vector cosine similarity and BM25 assign it a near-identical score to the primary product.

### Case 3: Irrelevant (I) Brand Collision (Rank = 1)
- **Query ID / Text**: `swan pillow`
- **Top-Ranked Product**: `ROYAL SWAN Bed Pillows for Sleeping 2 Pack,Best Neck Support`
- **Gold Label**: `I` (Irrelevant)
- **Why it failed**: The product is a regular rectangular neck pillow manufactured by the brand "ROYAL SWAN". Lexical matching matched the brand token "swan", while the user intent was a novelty swan-shaped pillow.

### Case 4: Category Keyword Collision (Rank = 1)
- **Query ID / Text**: `weights for exercises`
- **Top-Ranked Product**: `Vive Body Weight Workout Poster - Bodyweight Exercises For Home Gym`
- **Gold Label**: `C` (Complement)
- **Why it failed**: The user searched for physical weights (dumbbells/kettlebells), but the retrieved top result was an exercise poster containing the words "Body Weight Exercises".

### Case 5: Partial Lexical Match on Common Names (Rank = 1)
- **Query ID / Text**: `boggs`
- **Top-Ranked Product**: `The Gang Beats Boggs` (It's Always Sunny in Philadelphia media)
- **Gold Label**: `I` (Irrelevant)
- **Why it failed**: Ambiguous short query matched a specific TV show episode title rather than the intended baseball or outdoor gear brand.

---

## 4. Architectural Mitigations

1. **Cross-Encoder Reranking (`+rerank`)**:
   - Ingests full query and product context jointly in attention layers.
   - Boosts NDCG@10 from `0.5132` (Hybrid) to `0.5810`, effectively pushing companion guides and brand-collision items below true exact matches.
2. **Deterministic Pre-filtering & Category Gating**:
   - Extracting product categories during query understanding prevents accessories/posters from occupying primary item candidate slots.
