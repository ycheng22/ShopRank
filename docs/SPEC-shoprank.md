# SPEC-shoprank.md — ShopRank 模块规格（P1）

> 本文件是给 AI 编码代理读的模块规格。全局硬约束见仓库根目录 `AGENTS.md`；
> 本文件只描述"这个模块具体长什么样"。两者冲突时以 `AGENTS.md` 为准。
> 产出的代码、注释、README、commit 一律英文。

## 22.1 目标与成功标准

| 项 | 定义 |
| --- | --- |
| 目标 | 在 Amazon ESCI 人工标注基准上，构建一条每一层贡献都可复跑验证的多语言电商检索流水线，并把排序理由可视化 |
| 主指标 | NDCG@10（用 E/S/C/I 四级增益） |
| 副指标 | recall@50、MRR@10、P50/P95 延迟、每千次查询成本、索引构建耗时、索引内存占用 |
| 成功标准 | ① 五级消融表每一级相对上一级有可解释的变化（涨或跌都要能解释）② 三语 × 四配置交叉表完整 ③ 768 维相对 1536 维的 NDCG 损失 < 2 个百分点 ④ P95 < 800 ms（缓存命中除外）⑤ 任何人 clone 后按 README 能复跑出同样的数字 |
| 非目标 | 不做用户画像与个性化、不做点击日志建模、不做真实交易与库存、不追求 SOTA 排行榜名次 |

## 22.2 数据与分层抽样设计

这一节是整个项目最需要人工决策的地方，**配比已冻结，不要重新设计**。全随机抽样会让每一级提升被稀释到看不出来，因为大部分查询用 BM25 就能答对，必须刻意提高困难样本的比例。

| 分层维度 | 分箱 | 目标配比与理由 |
| --- | --- | --- |
| 查询长度 | 短（1–2 词）/ 中（3–5）/ 长（6+） | **25 / 45 / 30**。长查询是稠密向量优势区，短查询是 BM25 优势区，两端都要有足够样本才能画出对比 |
| 标注难度 | 易（E 占比高）/ 难（S、C、I 占比高） | **40 / 60**。刻意超采困难样本——这是消融表能显示出提升的关键 |
| 互补品存在性 | 含 C 标注 / 不含 | **至少 20% 含 C**。"搜手机壳返回手机"是电商检索最典型的错误，要有专门样本能测它 |
| 候选集规模 | 小（<20 候选）/ 大（≥20） | **40 / 60**。大候选集才体现重排价值 |
| 语言 | 英语原生 / 中文翻译 / 法语翻译 | 主实验集全英语（`locale = us`）；另建两个平行的翻译查询集，商品库始终为英语（这样测的是跨语言检索） |

**规模与切分**

- 查询 **4,000** 条（train 2,400 / dev 800 / test 800）。
- 商品取这些查询的全部候选商品去重，**上限 30,000 条**（`--max-products`，默认 30000）。上限触顶时**整条查询丢弃，绝不截断某个查询的候选集**——截断会破坏 qrels 完整性，让 recall 无端偏低；脚本须报告丢弃了多少条查询。
- 起步 30k 的理由：0.5 GB 免费档要同时装下 768 维向量、HNSW 索引与 tsv 全文索引。索引建好后实测占用，确认有余量再考虑扩到 5 万，并在 `docs/DECISIONS.md` 留痕。
- 切分 **60 / 20 / 20**，在上述五个维度上分层，**按 query 切而非按候选行切**，防止同一查询的候选跨切分泄漏。
- 随机种子 **seed=42**，硬编码并记入日志。抽样脚本必须提交入库。

**产物**

- `data/splits/{train,dev,test}.parquet`：query_id、text、locale、五个分层标签
- `data/products.parquet`：含 `product_text`（title + description 拼接，**嵌入与 tsv 索引共用同一列**）
- `data/qrels.parquet`：query_id、product_id、esci_label

**指标口径（写死在 `evals/metrics.py`，不要参数化）**

- 增益映射：**Exact = 3、Substitute = 2、Complement = 1、Irrelevant = 0**。必须在 README 中明示，因为不同论文取值不同。
- "相关"（供 recall 与 MRR 用）：**gain > 0**。
- qrels 为空的查询：**跳过，不计入均值**，并在每次跑分时报告跳过条数。若记 0 计入均值，长尾查询会把整体分数系统性拉低，且这种偏差在消融表上看不出来。

**翻译集构造**：用低价 provider 把 dev/test 的英语查询译为中文与法语，保留原 `query_id` 以便直接复用 qrels。README 明确标注为合成数据，仅用于跨语言召回评估，不用于报告绝对排序质量。

## 22.3 目录结构

| 路径 | 内容 |
| --- | --- |
| `app/main.py`、`app/routes/`、`app/deps.py` | FastAPI 入口、路由、依赖注入。不含业务逻辑 |
| `app/settings.py` | **全仓库唯一读取环境变量的地方**，产出一个 typed `Settings` 对象。其余模块通过参数接收配置 |
| `core/models.py` | 全部 Pydantic 领域模型（Query、Product、ScoredHit、SearchResponse、ScoreBreakdown、PipelineConfig） |
| `core/retrievers/` | `bm25.py`、`dense.py`、`base.py`（Retriever 协议） |
| `core/fusion.py` | RRF 与加权归一化融合 |
| `core/rerank.py` | cross-encoder 重排 |
| `core/query_understanding.py` | 纠错、属性抽取、意图分类（可裁剪模块） |
| `core/pipeline.py` | 唯一的编排入口，按配置组装各级，返回带分数明细的结果 |
| `core/embeddings.py`、`core/cache.py` | 嵌入抽象（含 768 维截断）、语义缓存 |
| `providers/` | LLM provider 抽象与路由（全项目唯一允许发起外部模型调用的地方） |
| `adapters/esci/` | ESCI 语料适配器：加载、清洗、归一化到领域模型 |
| `evals/` | `metrics.py`、`runner.py`、`ablation.py`、`configs/*.yaml`、`results/*.json` |
| `scripts/` | `sample_dataset.py`、`build_index.py`、`translate_queries.py`、`warm_cache.py`、`bench_batch_size.py`、`schema.sql`。**配置一律走命令行参数，不读环境变量** |
| `web/` | Angular 搜索界面与可解释面板。构建为静态产物，托管在 Cloudflare Pages |
| `docs/` | `ARCHITECTURE.md`、`DECISIONS.md`、`SPEC-*.md`、`LIMITATIONS.md`、`PROMPTS-W1.md` |

## 22.4 模块边界与接口定义

核心设计约束：**每一级都必须能被单独关闭，并且返回自己的原始分数**。这既是消融实验的前提，也是可解释面板的数据来源——两个需求由同一个设计满足。下面的签名请原样实现，不要改写。

| 模块 | 接口契约 |
| --- | --- |
| **Retriever 协议**<br>`core/retrievers/base.py` | `retrieve(query: Query, top_k: int) -> list[ScoredHit]`。`ScoredHit` 含 `product_id`、`raw_score`、`retriever_name`、`rank`。必须是 `typing.Protocol` 而非抽象基类——实现方**结构化满足即可，不需要继承**。BM25 与 Dense 实现同一协议，可互换、可并行调用 |
| **Embedder**<br>`core/embeddings.py` | `embed_documents(texts, dim) -> np.ndarray`、`embed_query(text, dim) -> np.ndarray`。`dim` 必须是参数而非常量，否则做不了维度消融表。落盘缓存的键为 `(model, dim, text hash)`——**dim 必须在键里**，否则 512/1536 维的跑分会静默复用 768 维向量 |
| **Fusion**<br>`core/fusion.py` | `fuse(runs: dict[str, list[ScoredHit]], method: Literal["rrf","weighted"], params) -> list[ScoredHit]`。输出必须保留各路原始分数，不能只留融合后的一个数。两路检索取相同的 `top_k`，避免融合被召回深度支配 |
| **Reranker**<br>`core/rerank.py` | `rerank(query, hits, top_k) -> list[ScoredHit]`。追加 `rerank_score`，保留重排前的名次（用于可解释面板展示"从第 12 名升到第 2 名"） |
| **Pipeline**<br>`core/pipeline.py` | `search(query: Query, config: PipelineConfig) -> SearchResponse`。`PipelineConfig` 是 Pydantic 模型，含 `use_bm25` / `use_dense` / `fusion_method` / `use_rerank` / `embed_dim` / `top_k` 等开关。**消融实验就是遍历这个 config，不需要改任何代码** |
| **ScoreBreakdown**<br>`core/models.py` | 每个结果附带 `bm25_score`、`dense_score`、`fused_score`、`rerank_score`、`rank_before_rerank`、`matched_terms`。各路分数是**累加而非覆盖**：融合分绝不能抹掉它由之计算而来的原始分。这就是可解释面板的完整数据契约 |
| **CorpusAdapter**<br>`adapters/base.py` | `iter_documents() -> Iterator[Product]`、`load_qrels() -> Qrels`。同样是 `typing.Protocol`。P2 与未来的语料接入必须实现同一协议——这是"同一内核多语料域"叙事的技术基础 |
| **Eval runner**<br>`evals/runner.py` | `run(config: PipelineConfig, split: str, dataset_version: str) -> EvalResult`。结果写入 `evals/results/` 与 `eval_runs` 表，含配置快照、数据集版本、指标、耗时、成本。**读 test 切分必须显式传 `allow_test=True`，否则抛异常** |

## 22.5 数据库与 API 契约

### 22.5.1 部署拓扑（决定下面所有 API 设计）

前端与后端**不同源**：

- 前端 `https://shoprank.vectorlab.me` — Angular 静态产物，Cloudflare Pages
- 后端 `https://shoprank-xxxxx.<region>.run.app` — FastAPI，Cloud Run，**无自定义域**

因此每一次浏览器调用都是跨源请求。CORS 与 HTTP 方法的选择不是风格问题，而是**冷启动体验与可用性问题**。

### 22.5.2 端点契约

| 端点 | 契约 |
| --- | --- |
| **`GET /api/search`** | 入参 `?q=&locale=`。**只接受预置示例查询**；未命中预置集合返回 400 并提示改用 POST。出参同 `SearchResponse`。响应头设 `Cache-Control: public, max-age=300, s-maxage=86400`。<br>选 GET 有三个叠加的理由：① GET + 标准头属 CORS **简单请求，没有 preflight**；② 可被 CDN 边缘缓存，**即使 Cloud Run 正在冷启动或完全挂掉，示例查询仍能秒开**；③ 与 `query_cache` 表构成同一思路的两层缓存 |
| **`POST /api/search`** | 入参 `{query, locale, config?}`；出参 `SearchResponse{hits[], breakdown[], latency_ms, cache_hit, config_used}`。自由输入路径，**需用户自带 API key**，不缓存。跨域 POST 带 JSON 必然先发一次 `OPTIONS`，冷启动时等于把延迟翻倍——所以这条路径**绝不承担"招聘方第一次点开"那个场景** |
| `GET /api/examples` | 返回 8 条预置示例查询（中英法各若干，刻意包含一条互补品陷阱查询和一条长尾查询）。可缓存 |
| `GET /api/ablation` | 从 `eval_runs` 读取并返回消融表 JSON，前端与 README 共用同一数据源。可缓存 |
| `GET /healthz` | 返回 `{"status":"ok","version":"<git sha>"}`。**绝不查询数据库**——没有 `SELECT 1`、没有连接池探测、没有迁移检查。保活任务每 10 分钟打一次，任何 DB 查询都会让 Neon compute 永不休眠、烧穿月度 CU-hours 并导致整个 project 被挂起 |

### 22.5.3 CORS

- `allow_origins` 是**显式白名单**，从 `Settings` 读取：`https://shoprank.vectorlab.me` 加上本地开发用的 `http://localhost:4200`。
- **禁止使用 `"*"`，开发环境也不例外。** 面试被问"你的公开 API 怎么防滥用"，显式白名单 + 限流是一个完整答案，`*` 是个扣分项。
- `allow_methods` 只开 `GET`、`POST`、`OPTIONS`；`allow_credentials` 保持关闭（本 API 不使用 cookie）。

### 22.5.4 前端契约

- 后端地址由「服务名-项目号-区域」决定，**换区域或换项目就会变**。前端必须把它作为**构建期环境变量注入**（Angular 的 `environment.prod.ts` 从 CI 变量读），**不得硬编码进源码**——这也正好符合 AGENTS.md §4"不得硬编码 endpoint"。
- 匿名访客只能点击预置示例（走 GET）。自由输入框在用户未提供 API key 时禁用，并提示原因——**不要返回错误页**。
- 所有请求带超时；后端不可用时降级为展示缓存的示例结果，而非白屏。

## 22.6 里程碑拆分（每个都是一次独立的 AI 任务）

| 里程碑 | 交付物 | 验收条件（达不到就不要进下一个） |
| --- | --- | --- |
| **M0a**<br>库骨架 | *在 `retrieval-core` 仓库*：领域模型、Retriever/CorpusAdapter 协议、pyproject、CI | ruff / mypy / pytest 全绿；依赖表干净；**没有 `app/`、`web/`、Dockerfile、`.env.example`**；全仓库搜不到 `os.environ`；不继承任何基类的假 retriever 能通过类型检查 |
| **M0b**<br>应用骨架与部署 | *在 `shoprank` 仓库*：FastAPI 起得来、`/healthz`、Docker Compose 起本地 Postgres、多阶段 Dockerfile、CI、部署到 Cloud Run | 公网 `*.run.app/healthz` 返回 200 且 sha 与最新提交一致；**故意失败的测试确实拦住了部署**；保活任务窗口外确实跳过 |
| **M1**<br>数据与切分 | 抽样脚本、三分切分、`schema.sql` | 脚本可复跑（两次结果字节级一致）；三个切分的分层分布偏差 < 2%；无跨切分泄漏；products / queries / qrels 三表落库 |
| **M2**<br>评测先行 | `metrics.py`、BM25 基线、`runner.py`、`ablation.py` | 指标有单元测试（用手工构造的小例子验证 NDCG 算对了）；BM25 基线数字进 `eval_runs` 与 README；README 的表由脚本生成 |
| **M3**<br>稠密检索与索引 | `embeddings.py`、`build_index.py`、`dense.py`、`fusion.py`、`pipeline.py` | 本地 GPU 批量嵌入完成并落盘缓存（删表重建不重算）；768 维入库且仍在免费档内；`EXPLAIN ANALYZE` 确认走 Index Scan；+dense / +hybrid 两行数字产出；`ScoreBreakdown` 保留各路原始分 |
| **M4**<br>融合与重排 | RRF 与加权两种融合、cross-encoder、难负例挖掘 | 五级消融表完整；难负例清单产出 |
| **M5**<br>多语言与跨语言 | 中法翻译集、跨语言检索通路 | 三语 × 四配置交叉表完整；每种语言各 20 条坏样本已人工审查并写进 `LIMITATIONS.md` |
| **M6**<br>UI 与上线 | Angular 界面 + 可解释面板 | 8 条预置示例走 GET 且命中缓存；限流与降级可用；GIF 录制；README 四张表齐全；前端部署在 Cloudflare Pages 并绑定域名 |

**M2 是整个项目的分水岭，不要跳过或延后。** 在没有基线数字之前写检索逻辑，等于失去了判断"我的改动到底有没有用"的能力——而这正是 AI 辅助开发最容易踩的坑：模型会很乐意生成一整套看起来专业的检索代码，但它无法告诉你这套代码是不是比一行 BM25 更好。那个判断只能来自你先建好的评测。