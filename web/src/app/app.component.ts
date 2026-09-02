import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../environments/environment';

interface ScoreBreakdown {
  bm25_score: number;
  dense_score: number;
  fused_score: number;
  rerank_score?: number;
  rank_before_rerank?: number;
}

interface Hit {
  product_id: string;
  title?: string;
  description?: string;
  product_text?: string;
  raw_score: number;
  breakdown: ScoreBreakdown;
}

interface SearchResponse {
  hits: Hit[];
  total_found: number;
  quota_exhausted?: boolean;
  error?: string;
  message?: string;
  examples_url?: string;
  is_preset_error?: boolean;
}

interface DemoQuery {
  query: string;
  locale: string;
  description: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'ShopRank';
  
  apiKey = '';
  searchQuery = '';
  locale = 'en';
  useDense = true;
  useRerank = false;
  fusionMethod = 'rrf';
  
  examples: DemoQuery[] = [];
  results: Hit[] = [];
  totalFound = 0;
  isSearching = false;
  quotaExhausted = false;
  demoModeMessage = '';
  presetError = '';
  
  expandedRow: number | null = null;
  
  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.loadExamples();
  }
  
  loadExamples() {
    this.http.get<DemoQuery[]>(`${environment.apiBaseUrl}/api/examples`).subscribe({
      next: (res) => this.examples = res,
      error: (err) => console.error("Failed to load examples", err)
    });
  }
  
  runPreset(example: DemoQuery) {
    this.searchQuery = example.query;
    this.locale = example.locale;
    
    this.isSearching = true;
    this.quotaExhausted = false;
    this.demoModeMessage = '';
    this.presetError = '';
    this.expandedRow = null;
    
    const params = {
      q: this.searchQuery,
      locale: this.locale,
      use_dense: this.useDense,
      use_rerank: this.useRerank,
      fusion_method: this.fusionMethod
    };
    
    this.http.get<SearchResponse>(`${environment.apiBaseUrl}/api/search`, { params }).subscribe({
      next: (res) => {
        this.results = res.hits || [];
        this.totalFound = res.total_found || 0;
        this.isSearching = false;
      },
      error: (err) => {
        this.isSearching = false;
        if (err.error && err.error.is_preset_error) {
           this.presetError = err.error.error;
           this.results = [];
        } else {
           console.error("Preset search failed", err);
           this.presetError = "Backend unreachable or preset not found. Check console.";
        }
      }
    });
  }
  
  onSearch() {
    if (!this.searchQuery.trim()) return;
    
    this.isSearching = true;
    this.quotaExhausted = false;
    this.demoModeMessage = '';
    this.presetError = '';
    this.expandedRow = null;
    
    const body = {
      query: this.searchQuery,
      locale: this.locale,
      config: {
        use_bm25: true,
        use_dense: this.useDense,
        use_rerank: this.useRerank,
        fusion_method: this.fusionMethod,
        embed_dim: 768,
        ef_search: 40,
        locale: this.locale
      }
    };
    
    const headers = new HttpHeaders({
      'X-Api-Key': this.apiKey || ''
    });
    
    this.http.post<SearchResponse>(`${environment.apiBaseUrl}/api/search`, body, { headers }).subscribe({
      next: (res) => {
        if (res.error === 'demo_mode') {
           this.demoModeMessage = res.message || '';
           this.results = [];
        } else {
           this.results = res.hits || [];
           this.totalFound = res.total_found || 0;
           this.quotaExhausted = res.quota_exhausted || false;
           if (this.quotaExhausted) {
              this.useRerank = false;
           }
        }
        this.isSearching = false;
      },
      error: (err) => {
        this.isSearching = false;
        console.error("Search failed", err);
      }
    });
  }
  
  toggleExpand(index: number) {
    if (this.expandedRow === index) {
      this.expandedRow = null;
    } else {
      this.expandedRow = index;
    }
  }
}
