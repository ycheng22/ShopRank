CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    locale TEXT NOT NULL,
    len_bin TEXT,
    difficulty_bin TEXT,
    has_complement BOOLEAN,
    candidate_size_bin TEXT,
    split TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_text TEXT NOT NULL,
    embedding vector(768),
    tsv tsvector
);

CREATE INDEX IF NOT EXISTS products_tsv_idx ON products USING GIN(tsv);
CREATE INDEX IF NOT EXISTS products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS qrels (
    query_id TEXT REFERENCES queries(query_id),
    product_id TEXT REFERENCES products(product_id),
    esci_label TEXT NOT NULL,
    PRIMARY KEY (query_id, product_id)
);
