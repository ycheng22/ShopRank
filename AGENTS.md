# AGENTS.md — Engineering Rules for AI Coding Agents

These rules are binding. If a rule blocks you, stop and report it — do not silently work around it.

## 0. Project context

- This is a portfolio of production-grade AI systems, not a prototype. Every module must be something you would defend in a senior engineering interview.
- The audience is technical hiring managers. Measured numbers, reproducibility, and honest documentation of limitations matter more than feature count.
- Hard monthly infrastructure budget: USD 20. Cost is a first-class design constraint, not an afterthought.
- Repos share one retrieval core. Prefer extending `retrieval-core` over duplicating logic in an application repo.
- The four repos are `retrieval-core`, `shoprank` (P1), `tripagent` (P2), `gatemark` (P3). `docsguard` (P4) is deferred and out of scope this month.

## 1. Locked technology choices — do not substitute

- Runtime: Python 3.12, async-first. Backend: FastAPI + Pydantic v2. Package manager: uv. Linter/formatter: ruff. Tests: pytest.
- Database and vector store: PostgreSQL + pgvector (Neon free tier). Do not introduce Qdrant, Pinecone, Chroma, Weaviate, Redis, Elasticsearch, or any managed vector service.
- Embeddings: BGE-M3, dense vectors truncated to 768 dimensions (Matryoshka). Reranking: bge-reranker-v2 locally, a small CPU cross-encoder in production.
- Never call a paid reranking API (Cohere, Voyage, Jina). This single item can break the budget.
- Agent framework: LangGraph. Frontend: Angular (ShopRank, TripAgent). Deployment: GCP Cloud Run. Bedrock is used only through the provider abstraction.
- All LLM calls go through `core/providers/`. No direct `openai.` / `anthropic.` / `boto3` calls anywhere else in the codebase.
- Adding any new runtime dependency requires an entry in `docs/DECISIONS.md` stating what it replaces and why.

### 1.1 Naming — frozen

The eval library is **GateMark**. GitHub repo `ycheng22/gatemark`, PyPI package `gatemark`, import `import gatemark`.

**Never use `evalgate` in any identifier, path, module, URL, or docstring.** `evalgate`, `evalgate-ci`, `evalgate-cli`, `evalgate-sdk`, and `llm-evalgate` are all taken on PyPI by unrelated projects with near-identical positioning.

### 1.2 Database topology — three Neon projects, one database each

| Neon project | Database | Contents |
| --- | --- | --- |
| `shoprank` | `shoprank` | ESCI products, 768-dim vectors, BM25 index |
| `tripagent` | `tripagent` | Wikivoyage / GTFS / OSM corpora and vectors |
| `gatemark` | `gatemark` | `eval_runs`, golden sets, prompt versions, cost attribution |

- **Never place two projects in one Neon project.** The free-tier 0.5 GB storage and 100 CU-hours are billed *per project*; splitting them yields three independent quotas.
- Run `CREATE EXTENSION IF NOT EXISTS vector;` **once per database**. Install into the `public` schema only. Do not create additional schemas — the `<->` operator fails to resolve outside `search_path`.
- For isolated dev/test data, create a **Neon branch** (10 per project, copy-on-write, no extra quota). **Never create a second database** and never suffix one with `_test`.
- Do not use the default `neondb` database name.
- **Storage budget:** stay within 0.5 GB per project. Start the ESCI sample at **30k products**, measure actual usage after the HNSW index is built, then decide whether to scale toward 50k. Once the limit is reached, Neon rejects *all* storage-increasing writes.

## 2. Evaluation comes first

- No retrieval or generation feature is implemented before its metric exists and a baseline number has been recorded. If asked to build a feature with no metric defined, stop and ask for the metric.
- Every experiment must be reproducible: fixed random seeds, sampling scripts committed, dataset version recorded alongside every result.
- The test split is used only for final reported numbers. Iterate on dev. Never tune against test.
- Deterministic metrics (NDCG, recall, latency) run in CI. LLM-as-judge evaluations never run in CI — they are triggered manually in batches to control cost.
- Never invent, estimate, round up, or placeholder a metric value. If a number is unknown, write `TBD`. Fabricated numbers are the single worst failure mode in this project.

## 3. Cost guardrails

- Anonymous visitors may only run pre-computed example queries served from cache. Free-form input requires a user-supplied API key.
- Enforce a per-day token quota in the application layer. On exhaustion, degrade to retrieval-only mode — never return an error page.
- All batch embedding runs use the provider's Batch endpoint and persist results to disk. Re-indexing must never re-pay for embeddings already computed.
- Every LLM call site must declare its purpose tag so token cost can be attributed by feature.
- Offline evaluation routes to low-cost providers; user-facing requests route to Western-hosted providers for latency and data-residency reasons. This choice is made in the provider layer, never at the call site.
- **`/healthz` MUST NOT touch the database.** It reports application liveness only — no `SELECT 1`, no connection-pool probe, no migration check. A keep-alive workflow polls it every 10 minutes for 12 hours a day; any DB query there keeps the Neon compute permanently awake and burns the monthly CU-hour quota, which suspends the whole project until the next billing cycle. This is the one place where the industry-standard pattern is wrong for this codebase.

## 4. Code standards

- Full type hints. mypy clean on `core/`. Pydantic models for every external boundary: HTTP request/response, tool arguments, config, dataset rows.
- Layering is strict: `app/` (HTTP) → `core/` (logic) → `adapters/` (corpora) → `providers/` (external calls). Dependencies point downward only. `core/` must never import from `app/`.
- No business logic in route handlers. No I/O in pure functions. No global mutable state.
- Configuration comes from environment variables via a single typed `Settings` object. No secrets, keys, endpoints, or absolute paths hard-coded anywhere. Update `.env.example` whenever a variable is added.
- Structured JSON logging with a request/trace id. Never `print()`. Never log secrets, full prompts containing user data, or raw API keys.
- Catch specific exceptions. Never use a bare `except:`. Every external call has an explicit timeout and a bounded retry with backoff.
- Tests are required for: retrieval logic, fusion and ranking math, metric implementations, tool argument validation, and every deterministic validator. UI and glue code may be untested.
- Commits follow Conventional Commits. Keep pull requests single-purpose.

## 5. Observability, security, and agent safety

- Instrument with OpenTelemetry GenAI semantic conventions from the first commit of a module, not retroactively. Every LLM and retrieval span records model, provider, token counts, latency, and purpose tag.
- Treat all retrieved content as untrusted input. Retrieved text is never concatenated into a system prompt and never interpreted as instructions.
- Agent tools are split into read and write. Read tools may execute automatically. Write tools require explicit approval, carry an idempotency key, and emit an audit record.
- Every agent run enforces three circuit breakers: maximum steps, maximum tokens, maximum monetary cost.
- Rate-limit every public endpoint. Validate and bound all user input lengths.

## 6. Documentation is part of the deliverable

- README first screen, in order: one-sentence value proposition, architecture diagram, metrics table, live demo link, 60-second GIF, cost constraint, Known Limitations.
- For `gatemark`, the README first screen must state `pip install gatemark`.
- A module is not done until `docs/DECISIONS.md` records any trade-off made while building it.
- Write documentation in English, plainly. Do not use marketing language, do not claim capabilities that are not measured, and never describe an unimplemented feature in the present tense.

## 7. Hard prohibitions

- Do not swap a locked dependency, add a new service, or change the database schema without being asked.
- **Do not use the name `evalgate` anywhere. The package is `gatemark`.**
- **Do not create a second database, an extra schema, or a second Neon project per application.**
- **Do not add any database access to `/healthz` or to the keep-alive workflow.**
- Do not modify files outside the paths named in the task.
- Do not delete or rewrite existing tests to make a build pass.
- Do not commit datasets, model weights, `.env` files, or credentials.
- Do not write placeholder metric values, fake benchmark results, or aspirational README claims.
- Do not call paid reranking APIs, and do not add always-on instances or paid tiers.
- Do not include the author's real name or personal email in code, commits, or package metadata — use the configured Git identity.
- Do not generate a large multi-module implementation in one pass. Build one module, make its tests pass, then stop and report.

## 8. Definition of done (every pull request)

1. `ruff`, `mypy`, and `pytest` pass locally and in CI.
2. New external boundaries have Pydantic models; new logic has tests.
3. Tracing spans emitted for any new LLM or retrieval path.
4. `.env.example` updated if configuration changed.
5. If behaviour affecting quality changed, the evaluation was re-run and the README metrics table updated with real numbers.
6. `docs/DECISIONS.md` updated for any deviation from these rules.
7. The PR description states what was built, what was measured, and what is still broken or unverified.