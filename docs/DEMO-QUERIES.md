# Preset Demo Queries

These 8 preset queries are designed to showcase different capabilities and failure modes of the ShopRank pipeline.

1. `wireless earbuds` (Locale: `en`)
   - **Type**: Short, high-traffic.
   - **Behavior**: Expect high relevance, highly competitive. Both BM25 and Dense do well.

2. `stainless steel insulated water bottle for hiking 32 oz` (Locale: `en`)
   - **Type**: Long, multi-attribute.
   - **Behavior**: Dense retrieval shines here by understanding the semantic constraints better than strict exact-match BM25.

3. `blouetooth speaker` (Locale: `en`)
   - **Type**: Misspelling.
   - **Behavior**: BM25 will likely fail entirely. Dense embeddings will map this close to "bluetooth speaker" and find the right products.

4. `蓝牙耳机` (Locale: `zh`)
   - **Type**: Cross-lingual (Chinese).
   - **Behavior**: Evaluates BGE-M3's ability to map Chinese queries to English product texts.

5. `casque sans fil` (Locale: `fr`)
   - **Type**: Cross-lingual (French).
   - **Behavior**: Evaluates BGE-M3's ability to map French queries to English product texts.

6. `iphone 14 pro` (Locale: `en`)
   - **Type**: Complement trap.
   - **Behavior**: Phones themselves vs phone cases/accessories. Reranker often needed to push actual phones to the top over highly-rated cases.

7. `silicone mold for resin jewelry making` (Locale: `en`)
   - **Type**: Long-tail, niche.
   - **Behavior**: Demonstrates retrieval performance on sparse, specific product categories.

8. `unicorn holographic laptop sticker pack aesthetic` (Locale: `en`)
   - **Type**: Deliberate failure.
   - **Behavior**: Likely sparse or no highly relevant results in the 30k sample. Demonstrates honest "no good match" behavior rather than surfacing unrelated junk.
