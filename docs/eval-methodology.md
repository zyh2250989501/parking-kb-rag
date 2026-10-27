# 评估方法

目标不是证明系统「能用」，而是回答三个问题：每一层（混合检索、重排、版本过滤）各自带来多少收益？错误主要出在检索还是生成？这些结论在题量有限时有多可信？

## 1. 题库构成

默认分层（`scripts/build_eval_set.py --per-type`）：

| 题型 | 数量 | 构造方式 | 考察点 |
| --- | --- | --- | --- |
| 事实查找 fact | 20 | 从单个 chunk 生成有唯一答案的问题 | 基础召回与生成 |
| 表格查询 table | 12 | 从含 markdown 表格的 chunk 生成数值题 | 表格切分与数值保真 |
| 跨文档 cross_doc | 10 | 同时引用两份文档 | 多路召回、答案整合 |
| 版本陷阱 version | 10 | 同 `doc_no` 新旧对；凑不出时用同 `doc_type` 两个版本，并标注 | 版本过滤是否选对现行版 |
| 不可答 unanswerable | 10 | 从语料随机采样但与知识库无关 | 拒答能力 |

共 62 题。题目与 gold 由 DeepSeek 生成，生成结果写入 `eval/questions.jsonl` 后**人工校对**：首轮 20 题，余力再补到 40 题；`EvalQuestion.verified` 标记是否已校对，报告按 verified / unverified / all 分组给出指标，避免未校对题目把结论抬高。

## 2. gold 如何锚定

`GoldSpan(doc_id, char_start, char_end, quote)`，字符区间相对于解析后的文档全文。

- 切分参数变化不影响标注有效性；评估时用 `resolve_gold_chunks` 找出与区间重叠的 chunk 作为 gold 集合。
- 若某题的区间在当次切分中找不到任何 chunk，该题记为「评估集问题」，不参与检索均值，并在失败案例里单列。
- 不可答题的 gold 为空，只参与拒答准确率。

## 3. 指标定义

设 `G` 为 gold chunk 集合，`R_k` 为检索返回的前 k 个 chunk。

| 指标 | 定义 |
| --- | --- |
| Recall@k | `|R_k ∩ G| / |G|` |
| MRR@10 | 第一个命中位置的倒数排名；未命中为 0 |
| nDCG@5 | `DCG@5 / IDCG@5`，折损 `1/log2(rank+2)`，二值相关性 |
| 引用精确率 | 答案引用的 chunk 中属于 gold 的比例 |
| 引用召回率 | gold chunk 被引用的比例 |
| 答案正确率 | judge 判为 `correct` 的比例（`partial` 单列） |
| Faithfulness | judge 判定答案中事实均可由上下文支撑的比例 |
| 拒答准确率 | `abstained` 与「是否为不可答题」一致的比例 |

统计口径：

- 检索与引用指标只在**有 gold 的可答题**上取均值；不可答题计入会把 Recall 人为拉低。
- 答案正确率 / Faithfulness / 拒答准确率统计全部题目。
- 括号内为 bootstrap 95% 置信区间：1000 次重采样，固定 seed=0，取 2.5% / 97.5% 分位。

## 4. 消融配置

| 配置 | 稠密 | BM25 | RRF | 重排 | 状态/版本过滤 |
| --- | --- | --- | --- | --- | --- |
| dense_only | ✓ | | | | |
| dense_bm25_rrf | ✓ | ✓ | ✓ | | |
| dense_bm25_rrf_rerank | ✓ | ✓ | ✓ | ✓ | |
| full | ✓ | ✓ | ✓ | ✓ | ✓ |

每组跑同一批题目、同一个生成模型与 judge。报告同时给出总表、分题型表与失败案例。

## 5. Judge

- 模型：SiliconFlow 上的 Qwen 系列（`JUDGE_MODEL`），与生成模型 DeepSeek 异源。
- 输出严格 JSON：`correct`（correct / partial / wrong）、`faithful`、`citation_precise`、`reason`。
- 解析失败或档位不在枚举内一律按 `wrong` 记录并写明原因，不做猜测映射——宁可少算，不要不可审计。
- 偏差控制：保留全部判定理由；在人工校对子集上另算 judge 与人工的一致率。该一致率数字待真实评估后补入报告。

## 6. 错误分类

每题只记一个主因，优先级：

`评估集问题` → `召回缺失` → `拒答错误` → `版本混淆`（仅 version 题）/ `生成错误` → `生成幻觉`。

单主因便于聚合计数；完整信号（检索排名、gold、判定理由、答案）都保留在 run record 里，需要多因分析时可重新归类。

## 7. 可复现契约

- 原始语料不入库（`corpus/*` 除来源清单与词典外均 gitignore）。
- `eval/questions.jsonl`、`eval/judge_verdicts.jsonl`、`reports/eval_report.md` 入库。
- `judge_verdicts.jsonl` 每条记录包含完整 run record（检索排名、gold、各项指标、判定），
  `python scripts/eval.py --report-only` 可在不联网、无 API key 的情况下重算并重写报告。
- 所有模型调用按内容哈希缓存，重跑同一批问题不会产生新的 API 调用。

## 8. 已知局限

- 题目由 LLM 生成，倾向「好问、好答」，指标可能偏乐观；靠人工校对与不可答题缓解，无法完全消除。
- 62 题的置信区间较宽，配置之间 2–3 个百分点的差距不足以支撑结论；报告中所有对比都附区间，跨区间才下判断。
- judge 与人工的一致性尚未量化，本次报告中的生成类指标应视为「judge 视角」的数字。
