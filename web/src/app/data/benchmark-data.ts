export interface AblationRow {
  configuration: string;
  ndcg10: number;
  recall50: number;
  mrr10: number;
  p95LatencyMs: number;
  deltaPercent: string;
  note: string;
}

export interface CrossLingualRow {
  locale: string;
  localeName: string;
  bm25: number;
  dense: number;
  hybrid: number;
  hybridRerank: number;
  gainComment: string;
}

export interface DimensionRow {
  dim: number;
  label: string;
  ndcg10: number;
  recall50: number;
  tableSizeMb: number;
  buildTimeSec: number;
  quotaNote: string;
}

export interface RrfSensitivityRow {
  kParam: string;
  kValue: number;
  ndcg10: number;
  recall50: number;
  mrr10: number;
  p95LatencyMs: number;
  behaviorNote: string;
}

export interface EngineeringMetricRow {
  metric: string;
  measuredValue: string;
  constraintTarget: string;
  status: 'PASS' | 'WARN';
  description: string;
}

export interface RelevanceConventionRow {
  code: string;
  label: string;
  gain: number;
  meaning: string;
  countedAsRelevant: boolean;
}

export const ABLATION_ROWS: AblationRow[] = [
  {
    configuration: 'BM25 Baseline',
    ndcg10: 0.4174,
    recall50: 0.5097,
    mrr10: 0.6616,
    p95LatencyMs: 172.4,
    deltaPercent: 'Baseline',
    note: 'PostgreSQL tsvector lexical exact match over title and product text'
  },
  {
    configuration: '+ Dense (BGE-M3 768d)',
    ndcg10: 0.4937,
    recall50: 0.6350,
    mrr10: 0.7234,
    p95LatencyMs: 275.2,
    deltaPercent: '+18.3%',
    note: 'pgvector HNSW index, ef_search=40; lifts Recall@50 by +12.5% abs'
  },
  {
    configuration: '+ Hybrid (RRF k=60)',
    ndcg10: 0.5132,
    recall50: 0.6323,
    mrr10: 0.7463,
    p95LatencyMs: 364.0,
    deltaPercent: '+23.0%',
    note: 'Zero-parameter reciprocal rank consensus; optimal speed/accuracy trade-off'
  },
  {
    configuration: '+ Cross-Encoder Rerank',
    ndcg10: 0.5810,
    recall50: 0.5706,
    mrr10: 0.7939,
    p95LatencyMs: 59302.2,
    deltaPercent: '+39.2%',
    note: 'BAAI/bge-reranker-v2-m3 on Top-30; peak accuracy, cached for edge delivery'
  },
  {
    configuration: '+ Weighted Fusion (0.5/0.5)',
    ndcg10: 0.5777,
    recall50: 0.5617,
    mrr10: 0.7896,
    p95LatencyMs: 58651.1,
    deltaPercent: '+38.4%',
    note: 'Linear score combination after min-max normalization; slightly trails RRF'
  }
];

export const CROSS_LINGUAL_ROWS: CrossLingualRow[] = [
  {
    locale: 'en',
    localeName: 'English (Native)',
    bm25: 0.4174,
    dense: 0.4937,
    hybrid: 0.5132,
    hybridRerank: 0.5810,
    gainComment: 'Native catalog baseline; both lexical and dense stages contribute'
  },
  {
    locale: 'zh',
    localeName: 'Chinese (Cross-lingual)',
    bm25: 0.0155,
    dense: 0.3013,
    hybrid: 0.2998,
    hybridRerank: 0.3821,
    gainComment: 'BM25 fails (0.0155); BGE-M3 bridges language gap to 0.3013 (+23.6x)'
  },
  {
    locale: 'fr',
    localeName: 'French (Cross-lingual)',
    bm25: 0.0975,
    dense: 0.3585,
    hybrid: 0.2710,
    hybridRerank: 0.3982,
    gainComment: 'BM25 catches few cognates; Dense + Rerank delivers 0.3982 (+3.1x)'
  }
];

export const DIMENSION_ROWS: DimensionRow[] = [
  {
    dim: 1024,
    label: 'Native BGE-M3',
    ndcg10: 0.5172,
    recall50: 0.6341,
    tableSizeMb: 486,
    buildTimeSec: 43,
    quotaNote: '486 MB (97.2% of Neon 500MB free quota) - at risk of write lock'
  },
  {
    dim: 768,
    label: 'Matryoshka Truncation (Chosen)',
    ndcg10: 0.5179,
    recall50: 0.6365,
    tableSizeMb: 418,
    buildTimeSec: 34,
    quotaNote: '418 MB (68 MB saved; regularizes tail noise; +0.0007 NDCG gain)'
  },
  {
    dim: 512,
    label: 'Heavy Truncation',
    ndcg10: 0.5119,
    recall50: 0.6299,
    tableSizeMb: 312,
    buildTimeSec: 15,
    quotaNote: '312 MB (Fastest build, but sacrifices -0.0060 NDCG)'
  }
];

export const RRF_SENSITIVITY_ROWS: RrfSensitivityRow[] = [
  {
    kParam: 'k = 10',
    kValue: 10,
    ndcg10: 0.5108,
    recall50: 0.6300,
    mrr10: 0.7431,
    p95LatencyMs: 340.2,
    behaviorNote: 'High top-rank bias; overweights rank #1 disagreement between retrievers'
  },
  {
    kParam: 'k = 60',
    kValue: 60,
    ndcg10: 0.5132,
    recall50: 0.6323,
    mrr10: 0.7463,
    p95LatencyMs: 364.0,
    behaviorNote: 'Optimal consensus balance; standard Cormack et al. benchmark configuration'
  },
  {
    kParam: 'k = 200',
    kValue: 200,
    ndcg10: 0.5118,
    recall50: 0.6323,
    mrr10: 0.7449,
    p95LatencyMs: 355.4,
    behaviorNote: 'Low rank decay; flattens score gradient, letting noisy tail ranks bleed in'
  }
];

export const ENGINEERING_METRIC_ROWS: EngineeringMetricRow[] = [
  {
    metric: 'Offline Index Build Time',
    measuredValue: '2,811 s (~46 min)',
    constraintTarget: 'Local GPU offline batch build',
    status: 'PASS',
    description: 'Embeds 29,844 products with BGE-M3 (768d) using PyTorch CUDA batching'
  },
  {
    metric: 'Peak GPU VRAM',
    measuredValue: '7,283 MiB',
    constraintTarget: 'RTX 2060 SUPER (8,192 MiB ceiling)',
    status: 'PASS',
    description: 'Managed via adaptive PyTorch token-bucketing with 0 out-of-memory events'
  },
  {
    metric: 'Database Disk Usage',
    measuredValue: '254 MB (post-vacuum)',
    constraintTarget: 'Neon Free Tier (500 MB quota)',
    status: 'PASS',
    description: '50.8% quota used, leaving 246 MB safety buffer below the hard write-lock'
  },
  {
    metric: 'Optimal Batch Size',
    measuredValue: '32 items (88.7 items/s)',
    constraintTarget: 'Memory-safe throughput plateau',
    status: 'PASS',
    description: 'Maximized CUDA core utilization without triggering memory swapping'
  },
  {
    metric: 'Online Hybrid P95 Latency',
    measuredValue: '364.0 ms',
    constraintTarget: 'Hard SLA: < 800 ms',
    status: 'PASS',
    description: 'Parallel PostgreSQL BM25 + pgvector HNSW query execution over connection pool'
  }
];

export const RELEVANCE_CONVENTIONS: RelevanceConventionRow[] = [
  {
    code: 'E',
    label: 'Exact',
    gain: 3,
    meaning: 'Product exactly fulfills the query specification, intent, and key constraints',
    countedAsRelevant: true
  },
  {
    code: 'S',
    label: 'Substitute',
    gain: 2,
    meaning: 'Alternative product with identical core utility (e.g. 24 oz vs 32 oz water bottle)',
    countedAsRelevant: true
  },
  {
    code: 'C',
    label: 'Complement',
    gain: 1,
    meaning: 'Accessory or secondary item frequently purchased with target (e.g. phone case for phone)',
    countedAsRelevant: true
  },
  {
    code: 'I',
    label: 'Irrelevant',
    gain: 0,
    meaning: 'Completely un-related or brand collision without functional relevance',
    countedAsRelevant: false
  }
];

export const KNOWN_LIMITATIONS = [
  {
    title: 'Cloud Run CPU Reranking Latency',
    description: 'Cross-encoder reranking requires ~59s on Cloud Run CPU instances. In production, free-form queries run on sub-400ms Hybrid retrieval, while heavy cross-encoders are cached at the CDN edge for preset examples.'
  },
  {
    title: 'Hardware Memory Guard on Cloud Run',
    description: 'Because large model weights and PyTorch dependencies exceed free-tier memory boundaries, live free-form search gracefully degrades to BM25 in cloud demo mode, while Dense and Reranking are precomputed for benchmark presets.'
  },
  {
    title: 'Machine-Translated Non-English Queries',
    description: 'Chinese and French query benchmarks are synthetically translated via DeepSeek and Qwen specifically to measure cross-lingual dense alignment against the fixed English product corpus.'
  }
];
