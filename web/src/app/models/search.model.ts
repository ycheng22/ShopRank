export type EsciLabel = 'Exact' | 'Substitute' | 'Complement' | 'Irrelevant';

export interface ScoreBreakdown {
  bm25_score: number;
  dense_score: number;
  fused_score: number;
  rerank_score?: number;
  rank_before_rerank?: number | null;
  matched_terms?: string[];
}

export interface Hit {
  product_id: string;
  title?: string;
  description?: string;
  product_text?: string;
  raw_score: number;
  breakdown: ScoreBreakdown;
  rank?: number;
  retriever_name?: string;
}

export interface SearchResponse {
  hits: Hit[];
  total_found: number;
  quota_exhausted?: boolean;
  error?: string;
  message?: string;
  examples_url?: string;
  is_preset_error?: boolean;
  degraded_to_bm25?: boolean;
}

export interface EsciResultDoc {
  asin: string;
  title: string;
  snippet: string;
  fullDescription: string;
  category: string;
  esciLabel: EsciLabel;
  bm25Rank: number;
  bm25RawScore: number;
  denseRank: number;
  denseCosine: number;
  rerankLogit: number; // Cross-encoder probability 0..1
  matchedTerms: string[];
  explainNote: string;
}

export interface PresetBenchmark {
  id: string;
  query: string;
  locale: 'en' | 'zh' | 'fr';
  localeDisplay: string;
  categoryNote: string;
  totalHits: number;
  latencyMs: { bm25: number; dense: number; rerank: number };
  docs: EsciResultDoc[];
}

export interface RankedItem extends EsciResultDoc {
  fusionScore: number;
  finalScore: number;
  formulaText: string;
  rankDelta: number;
}
