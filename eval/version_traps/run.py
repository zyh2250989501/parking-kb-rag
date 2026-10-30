"""版本陷阱实测：造一组同文号新旧版，走真实入库 → 建索引 → 检索。

公开语料里没有同一文号的新旧版本，也没有一份「失效」文档，所以设计文档 §9 的
「版本陷阱题选对现行版本」一行量不出数字。§9 允许在公开语料凑不出时构造，这里
就构造一组：同一个文号 长发改价费〔2019〕88 号 下挂三个版本，

- 2019 版：正文自述已废止 → 元数据判为「失效」
- 2023 版：元数据仍标「现行」（知识库里状态没来得及更新的真实情形）
- 2025 版：现行，生效日期最新

三份文档正文几乎逐字相同，只有收费数字不同，所以不做过期治理时，检索很容易
把旧版段落带进上下文。凡是「现行」的题，答案都必须落在 2025 版上。

用法：

    python eval/version_traps/run.py

产物写在 eval/version_traps/{corpus,data}/（已 gitignore），报告由 --report 指定。
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from docx import Document

from parking_kb.config import load_settings
from parking_kb.index.dense import BgeM3Embedder, DenseIndex
from parking_kb.index.sparse import SparseIndex
from parking_kb.index.store import ChunkStore, load_chunks_jsonl
from parking_kb.ingest.pipeline import run_ingest
from parking_kb.retrieve.hybrid import HybridRetriever, RetrievalConfig
from parking_kb.retrieve.rerank import BgeReranker
from parking_kb.types import DocumentMeta

DOC_NO = "长发改价费〔2019〕88 号"
_YEAR = re.compile(r"（(\d{4})年修订）")

# 三版只在收费数字上不同。year 同时用作文件名、标题与正文里的年份线索。
VERSIONS = [
    {
        "year": 2019,
        "effective": (2019, 5, 1),
        "free_minutes": "15",
        "first_hour": "8",
        "extra_hour": "5",
        "daily_cap": "60",
        "nightly": "350",
        "repealed": True,
    },
    {
        "year": 2023,
        "effective": (2023, 6, 1),
        "free_minutes": "20",
        "first_hour": "6",
        "extra_hour": "4",
        "daily_cap": "40",
        "nightly": "300",
        "repealed": False,
    },
    {
        "year": 2025,
        "effective": (2025, 1, 1),
        "free_minutes": "30",
        "first_hour": "5",
        "extra_hour": "3",
        "daily_cap": "30",
        "nightly": "240",
        "repealed": False,
    },
]

# 期望值全部取自 2025 版。
TRAPS = [
    ("长丰市机动车免费停放时长是多少分钟？", "30"),
    ("长丰市核心区路内停车泊位首小时收费多少元？", "5"),
    ("长丰市停车第二小时起每小时加收多少元？", "3"),
    ("长丰市停车单日最高收费是多少元？", "30"),
    ("长丰市夜间停车包月收费标准是多少元？", "240"),
]

# 带年份线索时不该过期过滤：2019 版要能被找回来。
TEMPORAL = ("2019年版的办法规定免费停放时长是多少分钟？", "15", 2019)


def _title(year: int) -> str:
    return f"长丰市机动车停放服务收费管理办法（{year}年修订）"


def build_docx(path: Path, spec: dict) -> None:
    year = spec["year"]
    ey, em, ed = spec["effective"]
    document = Document()
    document.add_heading(_title(year), level=1)
    document.add_paragraph(f"{DOC_NO}。")
    opened = f"本办法自{ey}年{em}月{ed}日起施行。"
    if spec["repealed"]:
        opened += "本办法已于2025年1月1日废止，由2025年修订版替代。"
    document.add_paragraph(opened)
    document.add_heading("一、适用范围", level=2)
    document.add_paragraph(
        "本办法适用于长丰市城区道路停车泊位、政府定价管理的公共停车场，"
        "以及火车站、汽车站配套停车场的停放服务收费。"
    )
    document.add_heading("二、免费停放时长", level=2)
    document.add_paragraph(
        f"机动车进入收费停车泊位后，免费停放时长为{spec['free_minutes']}分钟。"
        "停放时间超过免费时长的，按实际停放时间计费，免费时段不计入收费时间。"
    )
    document.add_heading("三、收费标准", level=2)
    document.add_paragraph(
        f"核心区路内停车泊位首小时收费{spec['first_hour']}元，"
        f"第二小时起每小时加收{spec['extra_hour']}元，单日最高收费{spec['daily_cap']}元。"
    )
    document.add_paragraph(
        f"夜间（20:00 至次日 8:00）停车包月收费标准为{spec['nightly']}元。"
    )
    document.add_heading("四、其他", level=2)
    document.add_paragraph(
        "执行任务的军车、警车、消防车、救护车免收停放服务费。"
        "本办法由长丰市发展和改革委员会负责解释。"
    )
    document.save(path)


def make_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for stale in corpus_dir.glob("*.docx"):
        stale.unlink()
    for spec in VERSIONS:
        path = corpus_dir / f"{_title(spec['year'])}.docx"
        build_docx(path, spec)


def _probe(retriever: HybridRetriever, question: str, config: RetrievalConfig):
    hits = retriever.retrieve(question, config)
    rows = [
        {
            "rank": rank,
            "doc": hit.chunk.meta.title,
            "section": hit.chunk.section_path,
            "text": hit.chunk.text,
        }
        for rank, hit in enumerate(hits, start=1)
    ]
    return rows


def _doc_year(title: str) -> int:
    match = _YEAR.search(title)
    return int(match.group(1)) if match else 0


def _score(rows: list[dict], question: str, kind: str, expected: str, wanted: int) -> dict:
    years = [_doc_year(row["doc"]) for row in rows]
    return {
        "question": question,
        "kind": kind,
        "expected": expected,
        # 期望值所在片段第一次出现的位置；「未命中」是 None。
        "hit_rank": next((row["rank"] for row in rows if expected in row["text"]), None),
        "years": years,
        # 陷阱题看上下文第一条是不是现行版；年份题看目标版本在不在上下文里。
        "correct": (years[0] == wanted) if kind == "trap" else wanted in years,
    }


def evaluate(retriever: HybridRetriever, config: RetrievalConfig) -> list[dict]:
    results = [
        _score(_probe(retriever, question, config), question, "trap", expected, 2025)
        for question, expected in TRAPS
    ]
    question, expected, year = TEMPORAL
    results.append(
        _score(_probe(retriever, question, config), question, "年份线索", expected, year)
    )
    return results


def _summary(results: dict[str, list[dict]]) -> str:
    lines = ["| 有效性 | 首条即现行版 | top-8 里没有旧版 |", "| --- | --- | --- |"]
    for name, items in results.items():
        traps = [item for item in items if item["kind"] == "trap"]
        right = sum(1 for item in traps if item["correct"])
        clean = sum(
            1 for item in traps if all(year == 2025 for year in item["years"])
        )
        lines.append(f"| {name} | {right}/{len(traps)} | {clean}/{len(traps)} |")
    return "\n".join(lines)


def build_report(documents: list[DocumentMeta], results: dict[str, list[dict]]) -> str:
    doc_rows = "\n".join(
        f"| {doc.title} | {doc.status} | {doc.effective_date} |" for doc in documents
    )
    detail = ["| 有效性 | 题型 | 问 | 期望 | 命中排名 | top-8 里的版本 |", "| --- | --- | --- | --- | --- | --- |"]
    for name, items in results.items():
        for item in items:
            rank = item["hit_rank"] if item["hit_rank"] is not None else "未命中"
            versions = "、".join(str(year) for year in item["years"]) or "空"
            detail.append(
                f"| {name} | {item['kind']} | {item['question']} | {item['expected']} "
                f"| {rank} | {versions} |"
            )
    temporal = results["full"][-1]
    return f"""# 版本陷阱实测（构造语料）

> **这不是公开语料上的成绩。** 公开语料（9 份标准/办法/手册）里没有同一文号的新旧
> 版本，也没有一份「失效」文档，设计文档 §9 的「版本陷阱题选对现行版本 ≥90%」在
> 原语料上量不出来。§9 允许凑不出时构造，本报告就是构造组的实测结果。
> 由 `python eval/version_traps/run.py` 生成，可原样重跑。

## 构造了什么

同一个文号 `长发改价费〔2019〕88 号` 下挂三个版本（脚本现场生成 DOCX，走真实解析 →
元数据抽取 → 切分 → 建索引 → 检索，全程没有覆写元数据）：

| 文档 | status | effective_date |
| --- | --- | --- |
{doc_rows}

三份正文除生效日期和五处数字（免费停放时长、首小时、每小时加收、单日上限、
夜间包月）外逐字相同。所以**语义检索本身没有任何线索**能选出对的那一份——
能区分的只有版本治理。

- 2019 版正文自述「本办法已于2025年1月1日废止」，元数据抽取判为 `失效`；
- 2023 版元数据仍是 `现行`，但同一文号下已有更新的一份——这是知识库里状态没来得及
  更新的真实情形，也是「同文号取最新生效日期」这条规则存在的理由；
- 2025 版 `现行`，生效日期最新。

## 题目

5 道版本陷阱题，期望值全部取自 2025 版；1 道年份线索题（「2019年版的办法规定免费
停放时长是多少分钟？」），期望值是 2019 版的 15 分钟。

## 结果

{_summary(results)}

逐题明细：

{chr(10).join(detail)}

## 怎么读

- **不做过期治理时，top-8 里 5/5 都混进了旧版**：三版正文几乎一样，重排给不出区分度，
  模型拿到的是三个互相矛盾的答案。首条正确的只有 1/5（20%）。
- **打开治理后 5/5 首条即现行版，top-8 里一条旧版都不剩**（100%）。版本选择和
  Recall 无关——治理发生在重排之前，被滤掉的版本根本进不了重排。
- **年份线索题按预期没有被过滤**：问句里的「2019年」触发时间线索，三版全部保留，
  2019 版的片段回到了上下文里（第 {temporal['hit_rank']} 位）。

## 局限

- **样本只有 5 道题。** 5/5 只能说明机制生效，不能说明「≥90%」这个阈值在真实语料上
  成立；真语料的难点在于版本对更多、元数据更脏。
- **年份线索题只保证「不过滤」，不保证「排在前面」。** 2019 版片段落在第 3 位：
  重排只看正文、不看标题，而三版正文除数字外完全相同，「2019」只出现在施行日期那一句里，
  重排无从判断。若生成阶段只用第一条，仍会答成 2025 版的 30 分钟。要修得把标题/生效
  日期拼进重排文本，那会改动全量评估的召回数字，未在本轮做。
- **构造的同文号并不常见**：现实里改版通常换新文号，那时能依靠的只有「失效硬过滤」和
  文档标题里的年份，本实验不能替代对真实企业语料的检验。
- **同分片段的先后顺序不稳**：三版正文除数字外相同，重排给同分片段的排序不保证每次
  一致。连跑三次，汇总数字（1/5、0/5、5/5、5/5）与 2019 版片段的位置（第 3 位）都没变，
  只有并列片段之间的次序互换。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="同文号新旧版的版本陷阱实测。")
    parser.add_argument("--root", type=Path, default=Path("eval/version_traps"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, default=Path("reports/version-traps.md"))
    args = parser.parse_args()

    corpus_dir = args.root / "corpus"
    data_dir = args.root / "data"
    make_corpus(corpus_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)

    report = run_ingest(corpus_dir, data_dir)
    if report.failed:
        raise SystemExit(f"入库失败：{report.failures}")
    print(f"入库 {report.succeeded}/{report.total} 份，{report.chunk_count} chunks")

    documents = [
        DocumentMeta(**json.loads(line))
        for line in (data_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for doc in documents:
        print(
            f"  {doc.title}: status={doc.status} "
            f"doc_no={doc.doc_no} effective_date={doc.effective_date}"
        )

    chunks = load_chunks_jsonl(data_dir / "chunks.jsonl")
    settings = load_settings(args.env if args.env.exists() else None)
    embedder = BgeM3Embedder(settings.embedding_model, device=args.device)
    dense = DenseIndex(data_dir / "qdrant")
    reranker = BgeReranker(settings.reranker_model, device=args.device)
    store = ChunkStore(data_dir / "kb.sqlite")
    try:
        dense.build(chunks, embedder)
        sparse = SparseIndex.build(chunks)
        store.upsert_documents(documents)
        store.upsert_chunks(chunks)
        retriever = HybridRetriever(dense, sparse, store, embedder, reranker)

        results = {}
        for name, config in (
            ("无状态过滤", RetrievalConfig(use_status_filter=False)),
            ("full", RetrievalConfig()),
        ):
            results[name] = evaluate(retriever, config)
            for item in results[name]:
                years = "、".join(str(year) for year in item["years"]) or "空"
                print(
                    f"[{name}] {item['kind']} {item['question']} "
                    f"期望={item['expected']} 排名={item['hit_rank']} 版本={years}"
                )
        print(_summary(results))

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(build_report(documents, results), encoding="utf-8")
        print(f"报告已写入 {args.report}")
    finally:
        dense.close()
        store.close()


if __name__ == "__main__":
    main()
