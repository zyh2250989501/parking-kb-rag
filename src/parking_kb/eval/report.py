import json
from collections import defaultdict
from pathlib import Path

TABLE_METRICS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@5",
    "citation_precision",
    "citation_recall",
    "answer_correct_rate",
    "faithful_rate",
    "abstention_accuracy",
)
HEADERS = {
    "recall@1": "Recall@1",
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "mrr@10": "MRR@10",
    "ndcg@5": "nDCG@5",
    "citation_precision": "引用精确率",
    "citation_recall": "引用召回率",
    "answer_correct_rate": "答案正确率",
    "faithful_rate": "Faithfulness",
    "abstention_accuracy": "拒答准确率",
}
TYPE_LABELS = {
    "fact": "事实查找",
    "table": "表格查询",
    "cross_doc": "跨文档",
    "version": "版本陷阱",
    "unanswerable": "不可答",
}


def _format(value, ci=None) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    text = f"{value:.3f}"
    if ci:
        text += f" ({ci[0]:.3f}–{ci[1]:.3f})"
    return text


def _ablation_table(aggregated: dict) -> list[str]:
    lines = [
        "| 配置 | " + " | ".join(HEADERS[key] for key in TABLE_METRICS) + " |",
        "| --- | " + " | ".join("---" for _ in TABLE_METRICS) + " |",
    ]
    for config_name, metrics in aggregated.items():
        cis = metrics.get("cis", {}) if isinstance(metrics, dict) else {}
        cells = [
            _format(metrics.get(key), cis.get(key)) for key in TABLE_METRICS
        ]
        lines.append(f"| {config_name} | " + " | ".join(cells) + " |")
    return lines


def _by_type_table(aggregated: dict) -> list[str]:
    lines = [
        "| 配置 | 题型 | 题数 | Recall@5 | 答案正确率 | 拒答准确率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    has_rows = False
    for config_name, metrics in aggregated.items():
        by_type = metrics.get("by_type", {}) if isinstance(metrics, dict) else {}
        for question_type, values in by_type.items():
            has_rows = True
            recall = (
                "—"
                if question_type == "unanswerable"
                else _format(values.get("recall@5"))
            )
            lines.append(
                f"| {config_name} | {TYPE_LABELS.get(question_type, question_type)} "
                f"| {values.get('count', 0)} | {recall} "
                f"| {_format(values.get('answer_correct_rate'))} "
                f"| {_format(values.get('abstention_accuracy'))} |"
            )
    return lines if has_rows else ["无分题型数据。"]


def _group_table(aggregated: dict) -> list[str]:
    lines = [
        "| 配置 | 分组 | 题数 | Recall@5 | 答案正确率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    has_rows = False
    for config_name, metrics in aggregated.items():
        groups = metrics.get("groups", {}) if isinstance(metrics, dict) else {}
        for group_name in ("verified", "unverified", "all"):
            values = groups.get(group_name)
            if not values:
                continue
            has_rows = True
            lines.append(
                f"| {config_name} | {group_name} | {values.get('count', 0)} "
                f"| {_format(values.get('recall@5'))} "
                f"| {_format(values.get('answer_correct_rate'))} |"
            )
    return lines if has_rows else ["无校对分组数据。"]


def _failure_section(verdicts: list[dict]) -> list[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for verdict in verdicts:
        error_type = verdict.get("error_type")
        if error_type:
            grouped[error_type].append(verdict)
    if not grouped:
        return ["无失败案例。"]

    lines: list[str] = []
    for error_type, items in grouped.items():
        lines.append(f"### {error_type}（{len(items)} 题，展示前 3 条）")
        lines.append("")
        for item in items[:3]:
            verdict = item.get("verdict") or {}
            lines.append(
                f"- **{item.get('config', '-')} / {item.get('qid', '-')}** "
                f"问题：{item.get('question', '')}"
            )
            lines.append(f"  - 答案：{item.get('answer', '')}")
            lines.append(
                f"  - 判定：{verdict.get('correct', item.get('correct', '-'))} "
                f"— {verdict.get('reason', item.get('reason', ''))}"
            )
        lines.append("")
    return lines


def render_eval_report(aggregated: dict, verdicts: list[dict]) -> str:
    lines = [
        "# RAG 评估报告",
        "",
        "指标说明：Recall/MRR/nDCG 只在有 gold 的可答题上统计；"
        "拒答准确率统计全部题目；括号内为 bootstrap 95% 置信区间。",
        "",
        "## 消融对比",
        "",
        *_ablation_table(aggregated),
        "",
        "## 分题型",
        "",
        *_by_type_table(aggregated),
        "",
        "## 人工校对分组",
        "",
        *_group_table(aggregated),
        "",
        "## 失败案例",
        "",
        *_failure_section(verdicts),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_judge_verdicts(verdicts: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(verdict, ensure_ascii=False) for verdict in verdicts]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_judge_verdicts(path: Path) -> list[dict]:
    verdicts: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                verdicts.append(json.loads(line))
    return verdicts
