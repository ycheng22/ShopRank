# Diagnostics & Failure Analysis

This document tracks known failure modes, hard negatives, and diagnostic analyses of the ShopRank retrieval pipeline.

## Known Hard Negatives (M4)

The following are top-ranked hard negatives (Rank=1) mined from the `us` locale dev set across three relevance categories (S/C/I). These examples represent cases where the BM25 + Dense hybrid retrieval currently struggles or misranks items.

### Substitute (S) Misrank
* **Query:** `the not a cat cat`
* **High-ranked Negative:** `I'm Here Live I'm Not A Cat: I'm Here Live I'm Not A Cat Fun`
* **Analysis:** The top result is a parody/merchandise item referencing a meme, rather than a genuine cat-related product. The exact word overlap ("Not A Cat") strongly triggers lexical retrieval.

### Complement (C) Misrank
* **Query:** `under a white sky: the nature of the future`
* **High-ranked Negative:** `Summary and Analysis of Under a White Sky: The Nature of the`
* **Analysis:** The top result is a study guide/summary rather than the original book. Because the titles are nearly identical, lexical and dense retrieval both score this extremely high.

### Irrelevant (I) Misrank
* **Query:** `swan pillow`
* **High-ranked Negative:** `ROYAL SWAN Bed Pillows for Sleeping 2 Pack,Best Neck Support`
* **Analysis:** The brand name contains "SWAN", which causes a lexical hit for "swan", but the product is a standard neck pillow, not a pillow shaped like a swan.
