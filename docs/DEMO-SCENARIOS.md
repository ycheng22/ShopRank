# Demo Scenarios

These 8 preset queries are hard-coded into the demo mode for users without an API key. They showcase ShopRank's handling of various e-commerce search challenges.

### 1. `wireless earbuds` (Locale: `en`)
- **Intent**: High-traffic, highly competitive exact-match query.
- **Expected Behavior**: Both BM25 and Dense do well. Reranking optimizes the exact rank order among highly relevant candidates.
- **Goal**: Show strong baseline performance.

### 2. `stainless steel insulated water bottle for hiking 32 oz` (Locale: `en`)
- **Intent**: Long, multi-attribute descriptive query.
- **Expected Behavior**: Dense retrieval shines here by understanding the semantic constraints (size, material, purpose) better than strict exact-match BM25 which gets confused by the length and rare token overlap.
- **Goal**: Demonstrate Dense embedding superiority for long-tail multi-attribute searches.

### 3. `blouetooth speaker` (Locale: `en`)
- **Intent**: Misspelled query.
- **Expected Behavior**: BM25 will likely fail entirely or return poor lexical matches. Dense embeddings map this close to "bluetooth speaker" in the latent space and find the correct products.
- **Goal**: Highlight robustness to typos without needing a dedicated spell-checker.

### 4. `蓝牙耳机` (Locale: `zh`)
- **Intent**: Cross-lingual (Chinese).
- **Expected Behavior**: Evaluates BGE-M3's ability to map Chinese queries to English product texts. BM25 will return 0 hits.
- **Goal**: Show native cross-lingual retrieval capability.

### 5. `casque sans fil` (Locale: `fr`)
- **Intent**: Cross-lingual (French).
- **Expected Behavior**: Same as Chinese, evaluates French mapping to English products.
- **Goal**: Show native cross-lingual retrieval capability.

### 6. `iphone 14 pro` (Locale: `en`)
- **Intent**: Complement trap.
- **Expected Behavior**: A classic problem where accessories (cases, screen protectors) dominate the lexical space because they repeatedly mention the target phone. Reranking is often needed to push actual phones to the top over highly-rated cases.
- **Goal**: Demonstrate the value of the cross-encoder in identifying central vs. complementary relevance.

### 7. `silicone mold for resin jewelry making` (Locale: `en`)
- **Intent**: Long-tail, niche.
- **Expected Behavior**: Demonstrates retrieval performance on sparse, specific product categories where only a few items might be relevant.
- **Goal**: Show long-tail competence.

### 8. `unicorn holographic laptop sticker pack aesthetic` (Locale: `en`)
- **Intent**: Deliberate failure / Extremely rare.
- **Expected Behavior**: Likely sparse or no highly relevant results in our 30k sample. 
- **Goal**: Demonstrates honest "no good match" behavior rather than surfacing unrelated junk to fill the page.
