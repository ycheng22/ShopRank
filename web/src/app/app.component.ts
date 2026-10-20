import { ChangeDetectionStrategy, Component, computed, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../environments/environment';
import {
  EsciResultDoc,
  Hit,
  PresetBenchmark,
  RankedItem,
  SearchResponse,
} from './models/search.model';
import { BENCHMARK_PRESETS } from './data/benchmark-presets';
import {
  ABLATION_ROWS,
  CROSS_LINGUAL_ROWS,
  DIMENSION_ROWS,
  ENGINEERING_METRIC_ROWS,
  KNOWN_LIMITATIONS,
  RELEVANCE_CONVENTIONS,
  RRF_SENSITIVITY_ROWS,
} from './data/benchmark-data';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'ShopRank';

  // Active top-level Tab
  activeTab = signal<'search' | 'benchmarks'>('search');

  // Form Controls
  searchControl = new FormControl<string>('wireless earbuds', { nonNullable: true });
  apiKeyControl = new FormControl<string>('', { nonNullable: true });

  // Pipeline Controls
  selectedLocale = signal<'en' | 'zh' | 'fr'>('en');
  pipelineDense = signal<boolean>(true);
  pipelineRerank = signal<boolean>(false);
  fusionMethod = signal<'RRF' | 'Weighted'>('RRF');

  // Explainability drawer open states
  expandedAsins = signal<Set<string>>(new Set());

  // All 8 Curated Presets
  readonly presets: PresetBenchmark[] = BENCHMARK_PRESETS;
  activePreset = signal<PresetBenchmark>(BENCHMARK_PRESETS[0]);

  // Backend Live State
  liveHits = signal<Hit[] | null>(null);
  liveTotalFound = signal<number | null>(null);
  isSearching = signal<boolean>(false);
  quotaExhausted = signal<boolean>(false);
  demoModeMessage = signal<string>('');
  presetError = signal<string>('');
  degradedMessage = signal<string>('');

  // Benchmark Reference Data for Tab 2
  readonly ablationRows = ABLATION_ROWS;
  readonly crossLingualRows = CROSS_LINGUAL_ROWS;
  readonly dimensionRows = DIMENSION_ROWS;
  readonly rrfRows = RRF_SENSITIVITY_ROWS;
  readonly engineeringRows = ENGINEERING_METRIC_ROWS;
  readonly relevanceConventions = RELEVANCE_CONVENTIONS;
  readonly knownLimitations = KNOWN_LIMITATIONS;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.autoExpandTopResult();
    // Attempt to warm up / fetch the initial preset from live API
    this.fetchPresetFromApi(this.activePreset());
  }

  activePipelineLabel = computed(() => {
    const stages = ['BM25'];
    if (this.pipelineDense()) stages.push('Dense');
    const base = stages.length > 1 ? `${stages.join(' + ')} (${this.fusionMethod()})` : 'BM25 Only';
    return this.pipelineRerank() ? `${base} → Cross-Encoder` : base;
  });

  totalLatencyMs = computed(() => {
    const l = this.activePreset().latencyMs;
    let sum = l.bm25;
    if (this.pipelineDense()) sum += l.dense;
    if (this.pipelineRerank()) sum += l.rerank;
    return sum.toFixed(1);
  });

  displayTotalHits = computed(() => {
    const live = this.liveTotalFound();
    if (live !== null && live !== undefined) {
      return live;
    }
    return this.activePreset().totalHits;
  });

  // Dynamically ranks & computes explainable formulas for all items
  rankedResults = computed<RankedItem[]>(() => {
    const preset = this.activePreset();
    const useDense = this.pipelineDense();
    const useRerank = this.pipelineRerank();
    const fusion = this.fusionMethod();
    const k = 60;

    const live = this.liveHits();

    // If live search returned hits, map them to explainable docs
    let baseDocs: EsciResultDoc[] = [];

    if (live && live.length > 0) {
      const presetDocsByAsin = new Map(preset.docs.map(d => [d.asin, d]));

      baseDocs = live.map((h, i) => {
        const cachedDoc = presetDocsByAsin.get(h.product_id);
        const bm25Raw = h.breakdown?.bm25_score || 0;
        const denseCosine = h.breakdown?.dense_score || 0;
        const rerankScore = h.breakdown?.rerank_score || 0;
        const matched = h.breakdown?.matched_terms || [];

        return {
          asin: h.product_id,
          title: h.title || cachedDoc?.title || `Product ${h.product_id}`,
          snippet: h.description || cachedDoc?.snippet || (h.product_text ? h.product_text.slice(0, 150) + '...' : ''),
          fullDescription: h.product_text || cachedDoc?.fullDescription || h.description || '',
          category: cachedDoc?.category || 'Catalog Product',
          esciLabel: cachedDoc?.esciLabel || (rerankScore > 0.8 ? 'Exact' : (rerankScore > 0.5 ? 'Substitute' : 'Irrelevant')),
          bm25Rank: cachedDoc?.bm25Rank || (i + 1),
          bm25RawScore: cachedDoc?.bm25RawScore || bm25Raw,
          denseRank: cachedDoc?.denseRank || (i + 1),
          denseCosine: cachedDoc?.denseCosine || denseCosine,
          rerankLogit: cachedDoc?.rerankLogit || rerankScore,
          matchedTerms: cachedDoc?.matchedTerms || matched,
          explainNote: cachedDoc?.explainNote || (h.retriever_name ? `Retrieved by ${h.retriever_name} engine` : 'Provenance computed via rank fusion')
        };
      });
    } else {
      baseDocs = preset.docs;
    }

    const evaluated = baseDocs.map(doc => {
      const rrfBm25 = 1 / (k + doc.bm25Rank);
      const rrfDense = 1 / (k + doc.denseRank);

      let fusionScore = rrfBm25;
      let formulaText = `1/(60+${doc.bm25Rank})`;

      if (useDense) {
        if (fusion === 'RRF') {
          fusionScore = rrfBm25 + rrfDense;
          formulaText = `1/(60+${doc.bm25Rank}) + 1/(60+${doc.denseRank})`;
        } else {
          // Weighted
          const normBm25 = Math.min(doc.bm25RawScore / 30, 1);
          fusionScore = 0.35 * normBm25 + 0.65 * doc.denseCosine;
          formulaText = `0.35*BM25_norm + 0.65*Cosine`;
        }
      }

      const finalScore = useRerank ? doc.rerankLogit : fusionScore;

      return {
        ...doc,
        fusionScore,
        finalScore,
        formulaText
      };
    });

    // Sort descending by finalScore
    evaluated.sort((a, b) => b.finalScore - a.finalScore);

    // Compute rank delta vs lexical BM25 order
    const bm25Order = [...baseDocs].sort((a, b) => a.bm25Rank - b.bm25Rank);

    return evaluated.map((item, idx) => {
      const baselineIdx = bm25Order.findIndex(d => d.asin === item.asin);
      const rankDelta = baselineIdx - idx;
      return {
        ...item,
        rankDelta
      };
    });
  });

  allExpanded = computed(() => {
    const list = this.rankedResults();
    const openSet = this.expandedAsins();
    return list.length > 0 && list.every(d => openSet.has(d.asin));
  });

  selectPreset(preset: PresetBenchmark): void {
    this.activePreset.set(preset);
    this.searchControl.setValue(preset.query);
    this.selectedLocale.set(preset.locale);

    // If dense was turned off solely due to live free-form search hardware degradation, restore it
    if (this.degradedMessage() && !this.pipelineDense()) {
      this.pipelineDense.set(true);
    }

    this.quotaExhausted.set(false);
    this.demoModeMessage.set('');
    this.presetError.set('');
    this.degradedMessage.set('');

    this.autoExpandTopResult();
    this.fetchPresetFromApi(preset);
  }

  fetchPresetFromApi(preset: PresetBenchmark): void {
    this.isSearching.set(true);

    const effectiveFusion = !this.pipelineDense() ? 'none' : this.fusionMethod().toLowerCase();
    const params = {
      q: preset.query,
      locale: preset.locale,
      use_dense: this.pipelineDense(),
      use_rerank: this.pipelineRerank(),
      fusion_method: effectiveFusion
    };

    this.http.get<SearchResponse>(`${environment.apiBaseUrl}/api/search`, { params }).subscribe({
      next: (res) => {
        this.isSearching.set(false);
        this.liveHits.set(res.hits || null);
        this.liveTotalFound.set(res.total_found || null);
        this.autoExpandTopResult();
      },
      error: (err) => {
        this.isSearching.set(false);
        const detail = err.error?.detail;
        if (detail && detail.is_preset_error) {
          this.presetError.set(detail.error);
        } else if (err.error && err.error.is_preset_error) {
          this.presetError.set(err.error.error);
        } else {
          // If offline or backend unreachable, fall back gracefully to local benchmark data
          console.warn("Serving preset from cached benchmark catalog", err);
          this.liveHits.set(null);
          this.liveTotalFound.set(preset.totalHits);
        }
        this.autoExpandTopResult();
      }
    });
  }

  runSearch(): void {
    const q = this.searchControl.value.trim();
    if (!q) return;

    // Check if query matches a known preset
    const match = this.presets.find(
      p => p.query.toLowerCase() === q.toLowerCase() || p.query.toLowerCase().includes(q.toLowerCase())
    );
    if (match && !this.apiKeyControl.value.trim()) {
      this.selectPreset(match);
      return;
    }

    // Free-form search via POST
    this.isSearching.set(true);
    this.quotaExhausted.set(false);
    this.demoModeMessage.set('');
    this.presetError.set('');
    this.degradedMessage.set('');

    const body = {
      query: q,
      locale: this.selectedLocale(),
      config: {
        use_bm25: true,
        use_dense: this.pipelineDense(),
        use_rerank: this.pipelineRerank(),
        fusion_method: this.fusionMethod().toLowerCase(),
        embed_dim: 768,
        ef_search: 40,
        locale: this.selectedLocale()
      }
    };

    const headers = new HttpHeaders({
      'X-Api-Key': this.apiKeyControl.value || ''
    });

    this.http.post<SearchResponse>(`${environment.apiBaseUrl}/api/search`, body, { headers }).subscribe({
      next: (res) => {
        this.isSearching.set(false);
        if (res.error === 'demo_mode') {
          this.demoModeMessage.set(res.message || '');
          this.liveHits.set([]);
          this.liveTotalFound.set(0);
        } else {
          this.liveHits.set(res.hits || []);
          this.liveTotalFound.set(res.total_found || 0);
          this.quotaExhausted.set(res.quota_exhausted || false);
          if (res.quota_exhausted) {
            this.pipelineRerank.set(false);
          }
          if (res.degraded_to_bm25) {
            this.degradedMessage.set(
              'Live free-form search ran on BM25 due to hardware limits (missing model weights). Dense + Cross-Encoder reranking can still be explored via the pre-computed demo presets above.'
            );
            this.pipelineDense.set(false);
            this.pipelineRerank.set(false);
          }
        }
        this.autoExpandTopResult();
      },
      error: (err) => {
        this.isSearching.set(false);
        console.error("Free-form search failed", err);
        this.presetError.set("Search request failed. Please check network connection.");
      }
    });
  }

  clearQuery(): void {
    this.searchControl.setValue('');
  }

  toggleDense(): void {
    this.pipelineDense.update(v => !v);
    this.refreshCurrentQuery();
  }

  toggleRerank(): void {
    this.pipelineRerank.update(v => !v);
    this.refreshCurrentQuery();
  }

  onLocaleChange(event: Event): void {
    const val = (event.target as HTMLSelectElement).value as 'en' | 'zh' | 'fr';
    this.selectedLocale.set(val);
    this.refreshCurrentQuery();
  }

  onFusionChange(event: Event): void {
    const val = (event.target as HTMLSelectElement).value as 'RRF' | 'Weighted';
    this.fusionMethod.set(val);
    this.refreshCurrentQuery();
  }

  private refreshCurrentQuery(): void {
    if (this.activePreset().query === this.searchControl.value) {
      this.fetchPresetFromApi(this.activePreset());
    } else {
      this.autoExpandTopResult();
    }
  }

  autoExpandTopResult(): void {
    queueMicrotask(() => {
      const results = this.rankedResults();
      if (results && results.length > 0) {
        this.expandedAsins.set(new Set([results[0].asin]));
      }
    });
  }

  isExpanded(asin: string): boolean {
    return this.expandedAsins().has(asin);
  }

  toggleExpand(asin: string): void {
    this.expandedAsins.update(prev => {
      const next = new Set(prev);
      if (next.has(asin)) {
        next.delete(asin);
      } else {
        next.add(asin);
      }
      return next;
    });
  }

  toggleAllDrawers(): void {
    const list = this.rankedResults();
    if (this.allExpanded()) {
      this.expandedAsins.set(new Set());
    } else {
      this.expandedAsins.set(new Set(list.map(d => d.asin)));
    }
  }
}
