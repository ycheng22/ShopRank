22.2 数据与分层抽样设计
这一节是整个项目最需要你亲自决策的地方。抽样设计决定了消融表好不好看——全随机抽样会让每一级提升都被稀释到看不出来，因为大部分查询用 BM25 就能答对。必须刻意提高困难样本的比例。
分层维度	分箱	目标配比与理由
查询长度	短（1–2 词）/ 中（3–5）/ 长（6+）	25 / 45 / 30。长查询是稠密向量优势区，短查询是 BM25 优势区，两端都要有足够样本才能画出对比
标注难度	易（E 占比高）/ 难（S、C、I 占比高）	40 / 60。刻意超采困难样本——这是消融表能显示出提升的关键
互补品存在性	含 C 标注 / 不含	至少 20% 含 C。"搜手机壳返回手机"是电商检索最典型的错误，要有专门的样本能测它
候选集规模	小（<20 候选）/ 大（≥20）	40 / 60。大候选集才体现重排价值
语言	英语原生 / 中文翻译 / 法语翻译	主实验集全英语；另建两个平行的翻译查询集，商品库始终为英语（这样测的是跨语言检索）
•	规模：查询 4,000 条（train 2,400 / dev 800 / test 800）；商品取这些查询的全部候选商品去重，预计 4–6 万条。切分按查询而非按候选行，防止同一查询的候选跨切分泄漏。
•	切分比例：60 / 20 / 20，按上述五个维度做分层切分，保证三个切分的分布一致。
•	产物：data/splits/{train,dev,test}.parquet（查询 id、查询文本、语言、分层标签）+ data/products.parquet + data/qrels.parquet（query_id、product_id、esci_label）。抽样脚本与随机种子（seed=42）必须提交入库。
•	增益映射（写死在 evals/metrics.py，不要让 AI 自己发挥）：Exact = 3、Substitute = 2、Complement = 1、Irrelevant = 0。这个映射必须在 README 中明示，因为不同论文取值不同，不写清楚数字就无法对照。
•	翻译集构造：用便宜模型把 dev/test 的英语查询译为中文与法语，保留原 query_id 以便直接复用 qrels。README 明确标注为合成数据，仅用于跨语言召回评估，不用于报告绝对排序质量。
22.3 目录结构
路径	内容
app/main.py、app/routes/、app/deps.py	FastAPI 入口、路由、依赖注入。不含业务逻辑
core/models.py	全部 Pydantic 领域模型（Query、Product、ScoredHit、SearchResponse、ScoreBreakdown）
core/retrievers/	bm25.py、dense.py、base.py（Retriever 协议）
core/fusion.py	RRF 与加权归一化融合
core/rerank.py	cross-encoder 重排
core/query_understanding.py	纠错、属性抽取、意图分类（可裁剪模块）
core/pipeline.py	唯一的编排入口，按配置组装各级，返回带分数明细的结果
core/embeddings.py、core/cache.py、core/settings.py	嵌入抽象（含 512 维截断）、语义缓存、typed Settings
providers/	LLM provider 抽象与路由（全项目唯一允许发起外部模型调用的地方）
adapters/esci/	ESCI 语料适配器：加载、清洗、归一化到领域模型
evals/	metrics.py、runner.py、ablation.py、configs/*.yaml、results/*.json
scripts/	sample_dataset.py、build_index.py、translate_queries.py、warm_cache.py
web/	Angular 搜索界面与可解释面板
docs/	ARCHITECTURE.md、DECISIONS.md、SPEC-*.md、LIMITATIONS.md
22.4 模块边界与接口定义
核心设计约束：每一级都必须能被单独关闭，并且返回自己的原始分数。这既是消融实验的前提，也是可解释面板的数据来源——两个需求由同一个设计满足。把下面的签名原样交给 AI，让它按此实现。
模块	接口契约
Retriever 协议
core/retrievers/base.py	retrieve(query: Query, top_k: int) -> list[ScoredHit]。ScoredHit 含 product_id、raw_score、retriever_name、rank。BM25 与 Dense 实现同一协议，可互换、可并行调用
Embedder
core/embeddings.py	embed_documents(texts, dim) -> np.ndarray、embed_query(text, dim) -> np.ndarray。dim 必须是参数而非常量，否则做不了维度消融表
Fusion
core/fusion.py	fuse(runs: dict[str, list[ScoredHit]], method: Literal["rrf","weighted"], params) -> list[ScoredHit]。输出必须保留各路原始分数，不能只留融合后的一个数
Reranker
core/rerank.py	rerank(query, hits, top_k) -> list[ScoredHit]。追加 rerank_score，保留重排前的名次（用于可解释面板展示"从第 12 名升到第 2 名"）
Pipeline
core/pipeline.py	search(query: Query, config: PipelineConfig) -> SearchResponse。PipelineConfig 是 Pydantic 模型，含 use_bm25 / use_dense / fusion_method / use_rerank / embed_dim / top_k 等开关。消融实验就是遍历这个 config，不需要改任何代码
ScoreBreakdown
core/models.py	每个结果附带：bm25_score、dense_score、fused_score、rerank_score、rank_before_rerank、matched_terms。这就是可解释面板的完整数据契约
CorpusAdapter
adapters/base.py	iter_documents() -> Iterator[Product]、load_qrels() -> Qrels。P2 与未来的语料接入必须实现同一协议——这是"同一内核多语料域"叙事的技术基础
Eval runner
evals/runner.py	run(config: PipelineConfig, split: str, dataset_version: str) -> EvalResult。结果写入 evals/results/，含配置快照、数据集版本、指标、耗时、成本
22.5 数据库与 API 契约
•	products：product_id (PK)、title、description、brand、color、locale、text_for_index（拼接后的索引文本）、embedding vector(512)、tsv tsvector（BM25/全文检索用）。索引：ivfflat 或 hnsw on embedding、GIN on tsv。
•	queries：query_id (PK)、text、locale、split、len_bin、difficulty_bin、has_complement、candidate_size_bin。分层标签直接存表，评测时可按维度切分报数。
•	qrels：query_id、product_id、esci_label、gain。
•	query_cache：cache_key、response_json、created_at。预置示例查询的完整结果预先写入这里——招聘方点开时命中缓存，即时返回、零成本，且所有 provider 挂掉时 demo 依然工作。
•	eval_runs：run_id、config_json、dataset_version、metrics_json、created_at。消融表直接从这张表生成，不要手工维护 Markdown 表格。
端点	契约
POST /api/search	入参 {query, locale, config?}；出参 SearchResponse{hits[], breakdown[], latency_ms, cache_hit, config_used}。匿名请求只允许 query 命中预置示例，自定义 config 需带 API key
GET /api/examples	返回 8 条预置示例查询（中英法各若干，刻意包含一条互补品陷阱查询和一条长尾查询）
GET /api/ablation	从 eval_runs 读取并返回消融表 JSON，前端与 README 共用同一数据源
GET /healthz	供 Cloud Run 与 GitHub Actions 保活任务探活（每天 12 小时窗口）
22.6 里程碑拆分（每个都是一次独立的 AI 任务）
里程碑	交付物	验收条件（达不到就不要进下一个）
M0	骨架与流水线	FastAPI 起得来、/healthz 通、Docker Compose 起本地 Postgres、CI 绿、部署到 Cloud Run 且域名可访问
M1	数据与切分	抽样脚本可复跑，三个切分的分层分布偏差 < 2%，products/queries/qrels 三表落库
M2	评测先行	evals/metrics.py 有单元测试（用手工构造的小例子验证 NDCG 算对了）；BM25 基线数字进 eval_runs 与 README
M3	稠密检索与索引	本地 GPU 批量嵌入完成并落盘缓存；512 维入库；+dense 一行数字产出；记录索引构建耗时与内存
M4	融合与重排	RRF 与加权两种融合可切换；cross-encoder 接入；五级消融表完整；难负例挖掘脚本产出困难样本清单
M5	多语言与跨语言	中法翻译集生成；三语 × 四配置交叉表完整；你亲自审查过每种语言各 20 条坏样本并写进 LIMITATIONS.md
M6	UI 与上线	Angular 界面 + 可解释面板；8 条预置示例结果落缓存；限流与降级；GIF 录制；README 四张表齐全
M2 是整个项目的分水岭，不要跳过或延后。在没有基线数字之前写检索逻辑，等于失去了判断"我的改动到底有没有用"的能力——而这正是 AI 辅助开发最容易踩的坑：模型会很乐意给你生成一整套看起来专业的检索代码，但它无法告诉你这套代码是不是比 BM25 更好。那个判断只能来自你先建好的评测。
