# 智慧停车企业知识库 RAG 问答助手 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 10 个工作日内建成一个可评估、可复现的停车企业知识库 RAG 问答助手，答案带出处、带生效日期，并用消融实验证明每个检索设计有效。

**Architecture:** 离线入库与在线检索严格分离。离线侧把异构文档解析为带元数据的结构化 chunk，建立向量索引、BM25 索引与 SQLite 元数据存储；在线侧做混合召回、RRF 融合、时效/版本过滤、重排与带引用生成。评估侧用分层评估集和异源 LLM judge 产出消融报告。

**Tech Stack:** Python 3.11、PyMuPDF、python-docx、openpyxl、RapidOCR、sentence-transformers（bge-m3 / bge-reranker）、Qdrant 本地模式、jieba + rank-bm25、SQLite、DeepSeek API、SiliconFlow API、FastAPI。

**Spec:** `docs/design/2026-10-26-parking-kb-rag-design.md`

## Global Constraints

- Python 3.11；Windows 11；无 Docker；GPU 为 RTX 4070 Laptop 8GB。
- 所有文档为中文，统一 UTF-8。
- 原始语料放 `corpus/`，生成产物放 `data/`，两者都在 `.gitignore` 中；仓库只提交 `corpus/sources.yaml`、`corpus/sources.md`、`corpus/metadata_overrides.yaml`、`corpus/jieba_userdict.txt`、评估题与最终判定缓存。
- API key 只能来自 `.env` 或环境变量；代码与文档中禁止出现密钥字面量。
- 生成模型：DeepSeek，模型名只从 `DEEPSEEK_MODEL` 读取，默认 `deepseek-chat`；temperature 0。
- Judge 模型：SiliconFlow 上的 Qwen 系列，模型名只从 `JUDGE_MODEL` 读取。
- 切分参数：目标 300~600 中文字符，最大 800，最小 100，重叠 100。
- 检索参数：稠密 top 20、稀疏 top 20、RRF k=60、融合 top 20、重排 top 8。
- 单元测试不得联网、不得调用真实 LLM/嵌入/重排；一律注入 Fake。
- 所有 LLM 调用必须经缓存层，缓存 key = hash(model + messages + params)。
- 提交信息使用 Conventional Commits，正文引用 `Spec: 2026-10-26-parking-kb-rag-design.md`。
- 时区固定 UTC+8；日期统一存 ISO `YYYY-MM-DD`。

## Review Focus

以下五类输入/场景在规格中没有被任何测试直接覆盖，但最可能让使用者踩坑。每个都在对应任务里加了测试：

1. **损坏或加密的 PDF**：单份文件失败不能让整次入库崩溃；必须记录失败原因并继续处理其余文件。
2. **无文本层的扫描 PDF**：必须路由到 OCR；不允许静默产出空 chunk 或整页跳过。
3. **Excel 合并单元格/多行表头**：合并区域的值必须向下/向右传播，表头不能丢，否则价目表检索会出错。
4. **询问已被替代的规则**：默认必须答现行版本并说明生效日期；只有用户明确问“旧标准/以前/某年”时才召回历史版本。
5. **语料中无答案的问题**：必须拒答并说明缺什么，不允许编造。

---

## 排期映射

| 天 | 任务 |
|----|------|
| 1 | Task 1、2、3（脚手架、语料下载、盘点） |
| 2 | Task 4、5、6（三类解析 + OCR 路由） |
| 3 | Task 7、8、9（清洗、元数据、切分） |
| 4 | Task 10、11、12（入库流水线、元数据存储、BM25） |
| 5 | Task 13、14、15（稠密索引、混合检索、版本过滤） |
| 6 | Task 16、17（重排、生成） |
| 7 | Task 18、19（指标、评估集构建与校对） |
| 8 | Task 20、21（Judge、评估运行器与报告） |
| 9 | 错误分析 + 针对性修复 + 重跑评估（复用 Task 21） |
| 10 | Task 22、23（API/前端、端到端与文档） |

---

## 文件结构

```text
parking-kb-rag/
├── pyproject.toml
├── .gitignore
├── .env.example
├── README.md
├── Makefile
├── corpus/
│   ├── sources.yaml
│   ├── sources.md
│   ├── metadata_overrides.yaml
│   └── jieba_userdict.txt
├── data/                              # gitignored
├── docs/
│   ├── design/2026-10-26-parking-kb-rag-design.md
│   ├── decisions.md
│   └── eval-methodology.md
├── prompts/
│   └── answer_v1.md
├── reports/                           # 提交小型 Markdown
├── scripts/
│   ├── fetch_corpus.py
│   ├── ingest.py
│   └── eval.py
├── src/parking_kb/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── ingest/
│   │   ├── scan.py
│   │   ├── clean.py
│   │   ├── metadata.py
│   │   ├── chunk.py
│   │   ├── pipeline.py
│   │   ├── report.py
│   │   └── parse/
│   │       ├── tables.py
│   │       ├── pdf.py
│   │       ├── docx.py
│   │       ├── xlsx.py
│   │       ├── ocr.py
│   │       └── registry.py
│   ├── index/
│   │   ├── store.py
│   │   ├── sparse.py
│   │   └── dense.py
│   ├── retrieve/
│   │   ├── hybrid.py
│   │   ├── filters.py
│   │   └── rerank.py
│   ├── generate/
│   │   ├── client.py
│   │   ├── prompt.py
│   │   └── schema.py
│   ├── eval/
│   │   ├── schema.py
│   │   ├── metrics.py
│   │   ├── build_set.py
│   │   ├── judge.py
│   │   ├── run.py
│   │   └── report.py
│   └── api/
│       ├── main.py
│       └── static/{index.html,app.js,styles.css}
└── tests/
    ├── conftest.py
    ├── fixtures/builders.py
    ├── ingest/
    ├── index/
    ├── retrieve/
    ├── generate/
    ├── eval/
    └── integration/
```

---

## Task 1: 项目脚手架、共享类型与配置

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `Makefile`, `src/parking_kb/__init__.py`, `src/parking_kb/types.py`, `src/parking_kb/config.py`
- Test: `tests/test_config.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Settings`（`config.py`）：`deepseek_api_key: str`、`deepseek_base_url: str`、`deepseek_model: str`、`siliconflow_api_key: str`、`siliconflow_base_url: str`、`judge_model: str`、`embedding_model: str`、`reranker_model: str`、`data_dir: Path`、`corpus_dir: Path`、`device: str`
  - `load_settings(env_file: Path | None = None) -> Settings`
  - `types.py` 中全部共享 dataclass：`FileInfo`、`Block`、`ParsedDoc`、`Section`、`DocumentMeta`、`Chunk`、`Candidate`、`Citation`、`AnswerResult`

- [ ] **Step 1: 初始化仓库与依赖**

```bash
cd "D:/VS工作/parking-kb-rag"
git init
```

`pyproject.toml` 依赖：`pydantic-settings>=2.0`、`pymupdf>=1.24`、`python-docx>=1.1`、`openpyxl>=3.1`、`sentence-transformers>=3.0`、`qdrant-client>=1.12`、`jieba>=0.42`、`rank-bm25>=0.2`、`openai>=1.40`、`fastapi>=0.115`、`uvicorn>=0.30`、`httpx>=0.27`、`pyyaml>=6.0`、`rapidocr-onnxruntime>=1.3`、`onnxruntime>=1.18`；dev 依赖 `pytest>=8.0`。

`.gitignore` 至少包含：`.env`、`corpus/*`（但保留 `corpus/sources.yaml`、`corpus/sources.md`、`corpus/metadata_overrides.yaml`、`corpus/jieba_userdict.txt`）、`data/`、`*.pyc`、`__pycache__/`、`.venv/`。

`tests/conftest.py` 提供后续任务共用的工厂函数：`file_info()`、`parsed_doc_with_text()`、`parsed_doc_with_headings()`、`parsed_doc_with_table()`、`meta()`、`chunk()`、`cand()`、`question()`、`answer()`。各任务直接使用这些名字，不重复造。

- [ ] **Step 2: 写失败测试**

```python
# tests/test_config.py
def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    s = load_settings(env_file=None)
    assert s.deepseek_model == "deepseek-chat"
    assert s.data_dir.name == "data"

def test_settings_requires_deepseek_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        load_settings(env_file=None)
```

- [ ] **Step 3: 运行并确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL，`ImportError` 或 `ValidationError` 未按预期触发。

- [ ] **Step 4: 实现 `Settings` 与 `load_settings`**

`Settings(BaseSettings)` 使用 `env_prefix=""`、`env_file=None` 由调用方决定；`deepseek_api_key` 必填；其余给默认值。`load_settings` 用 `Settings(_env_file=env_file)`。

同时在 `types.py` 定义共享 dataclass，字段如下：

```python
@dataclass
class FileInfo:
    path: Path; ext: str; size: int; sha256: str
    page_count: int | None = None; has_text_layer: bool | None = None
    text_char_count: int | None = None; is_scanned: bool = False
    error: str | None = None

@dataclass
class Block:
    kind: Literal["heading", "paragraph", "table", "list", "ocr_text"]
    text: str; level: int = 0; page: int | None = None

@dataclass
class ParsedDoc:
    doc_id: str; path: Path; title: str
    blocks: list[Block]; page_count: int
    warnings: list[str] = field(default_factory=list)

@dataclass
class Section:
    path: list[str]; blocks: list[Block]

@dataclass
class DocumentMeta:
    doc_id: str; title: str; doc_type: str; status: str
    doc_no: str | None = None; issuer: str | None = None
    effective_date: str | None = None; version: str | None = None
    source_url: str | None = None; page_count: int = 0

@dataclass
class Chunk:
    chunk_id: str; doc_id: str; text: str; section_path: str
    page_start: int | None; page_end: int | None; meta: DocumentMeta

@dataclass
class Candidate:
    chunk: Chunk; score: float
    dense_rank: int | None = None; sparse_rank: int | None = None

@dataclass
class Citation:
    n: int; chunk_id: str; doc_id: str; title: str
    effective_date: str | None; section_path: str
    page_start: int | None

@dataclass
class AnswerResult:
    answer: str; citations: list[Citation]; abstained: bool
    missing: list[str] = field(default_factory=list)
```

- [ ] **Step 5: 运行并确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml .gitignore .env.example Makefile src tests
git commit -m "chore: scaffold project, shared types, and settings

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 2: 公开语料下载与来源清单

**Files:**
- Create: `scripts/fetch_corpus.py`, `corpus/sources.yaml`, `corpus/sources.md`, `tests/test_fetch_corpus.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces:
  - `SourceEntry(name: str, url: str, doc_type: str, license_note: str)`
  - `load_sources(path: Path) -> list[SourceEntry]`（读取 YAML）
  - `render_sources_md(sources: list[SourceEntry]) -> str`
  - `download_sources(sources: list[SourceEntry], dest: Path, fetch: Callable[[str], bytes] = _http_get) -> list[Path]`
  - `_http_get(url: str) -> bytes`（httpx，10s 超时，浏览器 UA）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_fetch_corpus.py
def test_download_sources_writes_files(tmp_path):
    calls = []
    def fake_fetch(url):
        calls.append(url)
        return b"%PDF-1.4 test"
    sources = [SourceEntry("a.pdf", "https://example.com/a.pdf", "标准", "public")]
    paths = download_sources(sources, tmp_path, fetch=fake_fetch)
    assert paths == [tmp_path / "a.pdf"]
    assert calls == ["https://example.com/a.pdf"]

def test_download_sources_skips_existing(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"old")
    paths = download_sources([SourceEntry("a.pdf", "https://x", "标准", "public")], tmp_path)
    assert paths[0].read_bytes() == b"old"
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/test_fetch_corpus.py -v`
Expected: FAIL，`ImportError`。

- [ ] **Step 3: 实现下载器与来源清单**

`corpus/sources.yaml` 是机器可读清单，字段与 Appendix A 一一对应；`scripts/fetch_corpus.py` 另用 `render_sources_md` 生成给人看的 `corpus/sources.md`，避免同一份清单维护两遍。下载器只按 YAML 中的链接抓取；已存在同名文件时跳过；失败时打印 URL 与状态码但不中断其余下载。同时在 `corpus/sources.md` 标注需要手动补下的条目。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/test_fetch_corpus.py -v`
Expected: PASS。

- [ ] **Step 5: 手动抓取语料**

Run: `python scripts/fetch_corpus.py`
Expected: `corpus/` 下出现 PDF/DOCX/XLSX 文件；失败的 URL 手动补下，并在 `sources.md` 标注。

- [ ] **Step 6: 提交**

```bash
git add scripts/fetch_corpus.py corpus/sources.yaml corpus/sources.md tests/test_fetch_corpus.py .gitignore
git commit -m "feat: add public corpus fetcher and source manifest

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 3: 文件盘点与格式报告

**Files:**
- Create: `src/parking_kb/ingest/scan.py`, `tests/ingest/test_scan.py`, `tests/fixtures/builders.py`
- Test: `tests/ingest/test_scan.py`

**Interfaces:**
- Consumes: `FileInfo`、`Settings`
- Produces:
  - `scan_corpus(root: Path) -> list[FileInfo]`
  - `probe_pdf(path: Path) -> tuple[int, bool, int]`（页数、是否有文本层、可提取字符数）
  - `write_inventory_jsonl(files: list[FileInfo], path: Path) -> None`
  - `render_inventory_md(files: list[FileInfo]) -> str`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/test_scan.py
def test_scan_detects_scanned_pdf(tmp_path):
    make_image_only_pdf(tmp_path / "scan.pdf")
    info = scan_corpus(tmp_path)[0]
    assert info.ext == ".pdf"
    assert info.is_scanned is True
    assert info.has_text_layer is False

def test_scan_records_corrupt_file_and_continues(tmp_path):
    make_text_pdf(tmp_path / "ok.pdf")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    infos = scan_corpus(tmp_path)
    assert len(infos) == 2
    assert any(i.error for i in infos)
```

`tests/fixtures/builders.py` 用 PyMuPDF 生成两种 PDF：文本页；仅插入一张图片、无文本层的页。

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/test_scan.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现盘点器**

`probe_pdf` 用 `fitz.open`；逐页 `page.get_text("text")`，中位字符数 < 50 判为扫描件。扫描时对每个文件 `try/except`，异常写入 `FileInfo.error`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/test_scan.py -v`
Expected: PASS。

- [ ] **Step 5: 生成首份盘点报告**

Run: `python -c "from pathlib import Path; from parking_kb.ingest.scan import *; f=scan_corpus(Path('corpus')); Path('data').mkdir(exist_ok=True); write_inventory_jsonl(f, Path('data/inventory.jsonl')); Path('reports').mkdir(exist_ok=True); Path('reports/inventory.md').write_text(render_inventory_md(f), encoding='utf-8')"`
Expected: `reports/inventory.md` 列出格式分布、扫描件比例、损坏文件。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/ingest/scan.py tests/ingest/test_scan.py tests/fixtures/builders.py reports/inventory.md
git commit -m "feat: scan corpus and report format inventory

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 4: PDF 解析（文本、标题、表格）

**Files:**
- Create: `src/parking_kb/ingest/parse/tables.py`, `src/parking_kb/ingest/parse/pdf.py`, `tests/ingest/parse/test_pdf.py`
- Test: `tests/ingest/parse/test_pdf.py`

**Interfaces:**
- Consumes: `Block`、`ParsedDoc`
- Produces:
  - `ParseError(Exception)`
  - `table_to_markdown(rows: list[list[str | None]]) -> str`
  - `parse_pdf(path: Path, doc_id: str) -> ParsedDoc`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/parse/test_pdf.py
def test_parse_pdf_extracts_heading_and_table(tmp_path):
    p = make_mixed_pdf(tmp_path / "doc.pdf")   # 标题 + 段落 + 2x2 表格
    doc = parse_pdf(p, "d1")
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].text == "第一章 总则"
    assert any(b.kind == "table" and "| 车型 | 费率 |" in b.text for b in doc.blocks)

def test_parse_pdf_raises_parse_error_on_corrupt(tmp_path):
    p = tmp_path / "bad.pdf"; p.write_bytes(b"not a pdf")
    with pytest.raises(ParseError):
        parse_pdf(p, "bad")
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/parse/test_pdf.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现解析器**

用 PyMuPDF 的 `page.get_text("dict")` 取 span 的字号/加粗：字号显著大于正文中位数且加粗 → `heading`，`level` 按字号层级映射 1~3；否则 `paragraph`。表格用 `page.find_tables()`；`table_to_markdown` 单元值做 `strip()`，`None` 转空串，首行作为表头并补分隔行。解析异常统一抛 `ParseError(path, reason)`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/parse/test_pdf.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/ingest/parse/tables.py src/parking_kb/ingest/parse/pdf.py tests/ingest/parse/test_pdf.py
git commit -m "feat: parse PDF text, headings, and tables

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 5: DOCX 与 XLSX 解析

**Files:**
- Create: `src/parking_kb/ingest/parse/docx.py`, `src/parking_kb/ingest/parse/xlsx.py`, `tests/ingest/parse/test_docx.py`, `tests/ingest/parse/test_xlsx.py`
- Test: 同上

**Interfaces:**
- Consumes: `Block`、`ParsedDoc`、`table_to_markdown`
- Produces:
  - `parse_docx(path: Path, doc_id: str) -> ParsedDoc`
  - `parse_xlsx(path: Path, doc_id: str) -> ParsedDoc`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/parse/test_xlsx.py
def test_parse_xlsx_propagates_merged_cells(tmp_path):
    p = make_xlsx_with_merged_header(tmp_path / "prices.xlsx")
    doc = parse_xlsx(p, "x1")
    table = next(b for b in doc.blocks if b.kind == "table")
    assert "小型车" in table.text
    assert table.text.count("小型车") >= 2   # 合并单元格向右/向下传播
    assert any(b.kind == "heading" and "价目" in b.text for b in doc.blocks)

# tests/ingest/parse/test_docx.py
def test_parse_docx_uses_heading_styles(tmp_path):
    p = make_docx_with_headings(tmp_path / "policy.docx")
    doc = parse_docx(p, "w1")
    assert doc.blocks[0].kind == "heading"
    assert doc.blocks[0].level == 1
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/parse/test_docx.py tests/ingest/parse/test_xlsx.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现两个解析器**

DOCX：`Heading 1/2/3` 样式映射 `level`；表格转 Markdown。XLSX：每个 sheet 先产出一个 `heading` 块（sheet 名），再把区域转 Markdown；用 openpyxl 读取合并区域，把左上角值填充到区域内所有格子；首个非空行作为表头。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/parse/test_docx.py tests/ingest/parse/test_xlsx.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/ingest/parse/docx.py src/parking_kb/ingest/parse/xlsx.py tests/ingest/parse
git commit -m "feat: parse DOCX and XLSX with structure preservation

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 6: 扫描件 OCR 路由

**Files:**
- Create: `src/parking_kb/ingest/parse/ocr.py`, `src/parking_kb/ingest/parse/registry.py`, `tests/ingest/parse/test_ocr.py`
- Test: `tests/ingest/parse/test_ocr.py`

**Interfaces:**
- Consumes: `parse_pdf`、`parse_docx`、`parse_xlsx`、`FileInfo`
- Produces:
  - `OcrEngine` Protocol：`recognize(image_png: bytes) -> str`
  - `RapidOcrEngine()`：实现上述协议
  - `parse_scanned_pdf(path: Path, doc_id: str, engine: OcrEngine) -> ParsedDoc`
  - `parse_document(path: Path, doc_id: str, ocr_engine: OcrEngine | None = None) -> ParsedDoc`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/parse/test_ocr.py
class FakeOcr:
    def __init__(self, text): self.text = text
    def recognize(self, image_png): return self.text

def test_scanned_pdf_routes_to_ocr(tmp_path):
    p = make_image_only_pdf(tmp_path / "scan.pdf")
    doc = parse_document(p, "s1", ocr_engine=FakeOcr("停车收费标准"))
    assert doc.blocks[0].kind == "ocr_text"
    assert "停车收费标准" in doc.blocks[0].text

def test_text_pdf_never_calls_ocr(tmp_path):
    p = make_text_pdf(tmp_path / "text.pdf")
    class Boom:
        def recognize(self, _): raise AssertionError("OCR should not be called")
    doc = parse_document(p, "t1", ocr_engine=Boom())
    assert all(b.kind != "ocr_text" for b in doc.blocks)
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/parse/test_ocr.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现路由与 OCR 引擎**

`parse_document` 先按扩展名分发；PDF 先 `probe_pdf`，有文本层走 `parse_pdf`，否则走 `parse_scanned_pdf`。扫描页用 PyMuPDF 渲染 200 DPI PNG，交给 `RapidOcrEngine`；每页生成一个 `ocr_text` 块并记录页码。OCR 结果为空时写入 `ParsedDoc.warnings`，不静默丢弃。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/parse/test_ocr.py -v`
Expected: PASS。

- [ ] **Step 5: 真实 OCR 冒烟（可跳过）**

Run: `pytest tests/ingest/parse/test_ocr.py -v -m slow`
Expected: 若已安装 RapidOCR，真实扫描页能识别出非空文本；未安装则 skip。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/ingest/parse/ocr.py src/parking_kb/ingest/parse/registry.py tests/ingest/parse/test_ocr.py
git commit -m "feat: route scanned PDFs to OCR and never drop pages

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 7: 清洗（页眉页脚、换行、重复段落）

**Files:**
- Create: `src/parking_kb/ingest/clean.py`, `tests/ingest/test_clean.py`
- Test: `tests/ingest/test_clean.py`

**Interfaces:**
- Consumes: `Block`、`ParsedDoc`
- Produces:
  - `normalize_text(text: str) -> str`
  - `join_wrapped_lines(text: str) -> str`
  - `remove_repeated_lines(blocks: list[Block], page_count: int, threshold: float = 0.6) -> list[Block]`
  - `dedupe_blocks(blocks: list[Block], threshold: float = 0.9) -> list[Block]`
  - `clean_document(doc: ParsedDoc) -> ParsedDoc`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/test_clean.py
def test_remove_repeated_header_and_footer():
    blocks = []
    for page in (1, 2, 3):
        blocks += [Block("paragraph", "内部资料 严禁外传", page=page),
                   Block("paragraph", f"第 {page} 页", page=page),
                   Block("paragraph", f"正文{page}", page=page)]
    cleaned = remove_repeated_lines(blocks, page_count=3)
    assert [b.text for b in cleaned] == ["正文1", "正文2", "正文3"]

def test_join_wrapped_chinese_lines():
    raw = "停车收费标准按照车型\n和停放时长计算。"
    assert join_wrapped_lines(raw) == "停车收费标准按照车型和停放时长计算。"

def test_dedupe_near_duplicate_paragraphs():
    blocks = [Block("paragraph", "同一段话。" * 20), Block("paragraph", "同一段话。" * 20)]
    assert len(dedupe_blocks(blocks)) == 1
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/test_clean.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现清洗函数**

重复行判定：同一页仅出现一次、且至少在 `threshold × page_count` 个不同页出现、长度 < 40 的文本行。换行拼接：仅拼接同一段落内、上一行不以 `。！？；：` 结尾且下一行不以标题/列表符号开头的行。去重：字符 5-gram Jaccard > threshold 时保留先出现者。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/test_clean.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/ingest/clean.py tests/ingest/test_clean.py
git commit -m "feat: clean headers, footers, wrapped lines, and duplicates

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 8: 元数据抽取与人工覆盖

**Files:**
- Create: `src/parking_kb/ingest/metadata.py`, `corpus/metadata_overrides.yaml`, `tests/ingest/test_metadata.py`
- Test: `tests/ingest/test_metadata.py`

**Interfaces:**
- Consumes: `ParsedDoc`、`FileInfo`、`DocumentMeta`
- Produces:
  - `parse_doc_no(text: str) -> str | None`
  - `parse_effective_date(text: str) -> str | None`
  - `load_overrides(path: Path) -> dict[str, dict]`
  - `extract_metadata(doc: ParsedDoc, file_info: FileInfo, overrides: dict[str, dict]) -> DocumentMeta`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/test_metadata.py
def test_extract_doc_no_and_effective_date():
    doc = parsed_doc_with_text("湘发改价费规〔2025〕723号\n本办法自2026年1月1日起施行。")
    meta = extract_metadata(doc, file_info(), overrides={})
    assert meta.doc_no == "湘发改价费规〔2025〕723号"
    assert meta.effective_date == "2026-01-01"
    assert meta.status == "现行"

def test_override_wins_over_regex():
    doc = parsed_doc_with_text("某文件")
    overrides = {file_info().sha256: {"status": "失效", "effective_date": "2000-09-01"}}
    meta = extract_metadata(doc, file_info(), overrides)
    assert meta.status == "失效"
    assert meta.effective_date == "2000-09-01"

def test_unknown_status_when_no_signal():
    meta = extract_metadata(parsed_doc_with_text("普通通知"), file_info(), {})
    assert meta.status == "未知"
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/test_metadata.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现抽取**

文号正则：`[\u4e00-\u9fff]{1,12}〔\d{4}〕\d+号`。生效日期支持“自/于 YYYY 年 M 月 D 日起施行/执行/生效”。状态：正文含“废止/失效/已被替代” → `失效`；含“施行/生效” → `现行`；否则 `未知`。`overrides` 以文件 SHA-256 为 key，字段级覆盖。关键收费办法必须在 `metadata_overrides.yaml` 中显式标注。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/test_metadata.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/ingest/metadata.py corpus/metadata_overrides.yaml tests/ingest/test_metadata.py
git commit -m "feat: extract document metadata with manual overrides

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 9: 结构感知切分

**Files:**
- Create: `src/parking_kb/ingest/chunk.py`, `tests/ingest/test_chunk.py`
- Test: `tests/ingest/test_chunk.py`

**Interfaces:**
- Consumes: `ParsedDoc`、`DocumentMeta`、`Chunk`、`Section`
- Produces:
  - `ChunkConfig(target: int = 500, max: int = 800, min: int = 100, overlap: int = 100)`
  - `build_sections(blocks: list[Block]) -> list[Section]`
  - `split_text(text: str, config: ChunkConfig) -> list[str]`
  - `chunk_document(doc: ParsedDoc, meta: DocumentMeta, config: ChunkConfig = ChunkConfig()) -> list[Chunk]`
  - `make_chunk_id(doc_id: str, section_path: str, index: int) -> str`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/test_chunk.py
def test_chunk_carries_section_path_and_size_limit():
    doc = parsed_doc_with_headings()
    chunks = chunk_document(doc, meta(), ChunkConfig())
    assert chunks[0].section_path == "第一章 总则"
    assert all(len(c.text) <= 800 for c in chunks)

def test_long_section_splits_with_overlap():
    doc = parsed_doc_with_text("第一段。" * 400)
    chunks = chunk_document(doc, meta(), ChunkConfig())
    assert len(chunks) > 1
    assert chunks[0].text[-100:] in chunks[1].text

def test_table_is_atomic_and_keeps_header():
    doc = parsed_doc_with_table()
    chunk = next(c for c in chunk_document(doc, meta()) if "|" in c.text)
    assert "| 车型 |" in chunk.text
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/test_chunk.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现切分**

`build_sections` 维护标题栈：`heading` 更新栈，其余块归属当前栈路径；无标题时路径为空字符串。`split_text` 按 `。！？；\n` 切句，累加到 `target` 后成块；单块超过 `max` 强制切；相邻块保留 `overlap` 字符。表格块不拆（超过 200 行时按行组拆并重复表头）。小于 `min` 的块与同章节下一块合并。`chunk_id = sha1(f"{doc_id}|{section_path}|{index}")[:16]`。

`chunks.jsonl` 每行固定为 `{chunk_id, doc_id, text, section_path, page_start, page_end, meta: {完整 DocumentMeta 字段}}`；`documents.jsonl` 每行为一个 `DocumentMeta`。`load_chunks_jsonl` 按此结构反序列化。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/test_chunk.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/ingest/chunk.py tests/ingest/test_chunk.py
git commit -m "feat: structure-aware chunking with table preservation

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 10: 入库流水线、失败隔离与报告

**Files:**
- Create: `src/parking_kb/ingest/pipeline.py`, `src/parking_kb/ingest/report.py`, `scripts/ingest.py`, `tests/ingest/test_pipeline.py`
- Test: `tests/ingest/test_pipeline.py`

**Interfaces:**
- Consumes: Task 3~9 的全部函数
- Produces:
  - `IngestFailure(path: str, error: str)`
  - `IngestReport(total: int, succeeded: int, failed: int, failures: list[IngestFailure], document_count: int, chunk_count: int, scanned_docs: int, avg_chunk_len: float, missing_metadata: dict[str, int])`
  - `run_ingest(corpus_dir: Path, data_dir: Path, overrides_path: Path | None = None, ocr_engine: OcrEngine | None = None) -> IngestReport`
  - `write_report(report: IngestReport, path: Path) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/ingest/test_pipeline.py
def test_ingest_continues_after_corrupt_file(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)   # 2 正常 + 1 损坏
    report = run_ingest(corpus, data)
    assert report.succeeded == 2
    assert report.failed == 1
    assert "bad.pdf" in report.failures[0].path
    assert (data / "chunks.jsonl").exists()

def test_ingest_writes_report_with_failure_reason(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    report = run_ingest(corpus, data)
    write_report(report, data / "ingest_report.md")
    text = (data / "ingest_report.md").read_text(encoding="utf-8")
    assert "bad.pdf" in text and "失败原因" in text
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/ingest/test_pipeline.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现流水线与报告**

逐文件执行：scan → parse_document → clean_document → extract_metadata → chunk_document；每份文件 `try/except Exception`，失败写 `IngestFailure` 并继续。写出 `documents.jsonl`、`chunks.jsonl`、`ingest_report.md`。报告必须含：总数、成功/失败、失败清单与原因、扫描件数、chunk 数、平均长度、元数据缺失计数。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/ingest/test_pipeline.py -v`
Expected: PASS。

- [ ] **Step 5: 对真实语料跑一次**

Run: `python scripts/ingest.py --corpus corpus --data data`
Expected: `reports/ingest_report.md` 产出；失败文件逐条有原因；扫描件比例可见。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/ingest/pipeline.py src/parking_kb/ingest/report.py scripts/ingest.py tests/ingest/test_pipeline.py reports/ingest_report.md
git commit -m "feat: ingest pipeline with per-file failure isolation and report

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 11: 元数据存储（SQLite）

**Files:**
- Create: `src/parking_kb/index/store.py`, `tests/index/test_store.py`
- Test: `tests/index/test_store.py`

**Interfaces:**
- Consumes: `DocumentMeta`、`Chunk`
- Produces:
  - `ChunkStore(db_path: Path)`
  - `.upsert_documents(docs: list[DocumentMeta]) -> None`
  - `.upsert_chunks(chunks: list[Chunk]) -> None`
  - `.get_document(doc_id: str) -> DocumentMeta | None`
  - `.get_chunks(chunk_ids: list[str]) -> list[Chunk]`
  - `.get_chunks_by_doc(doc_id: str) -> list[Chunk]`
  - `.list_documents(status: str | None = None) -> list[DocumentMeta]`
  - `.find_versions(doc_no: str) -> list[DocumentMeta]`
  - `.close() -> None`
  - `load_chunks_jsonl(path: Path) -> list[Chunk]`

- [ ] **Step 1: 写失败测试**

```python
# tests/index/test_store.py
def test_store_roundtrip_and_version_lookup(tmp_path):
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents([meta(doc_id="d1", doc_no="X〔2025〕1号", effective_date="2026-01-01", status="现行")])
    store.upsert_chunks([chunk("c1", "d1", "收费按小时计")])
    assert store.get_document("d1").doc_no == "X〔2025〕1号"
    assert store.get_chunks(["c1"])[0].text == "收费按小时计"
    assert len(store.find_versions("X〔2025〕1号")) == 1

def test_list_documents_filters_by_status(tmp_path):
    store = ChunkStore(tmp_path / "kb.sqlite")
    store.upsert_documents([meta(doc_id="d1", status="现行"), meta(doc_id="d2", status="失效")])
    assert [d.doc_id for d in store.list_documents(status="现行")] == ["d1"]
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/index/test_store.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现存储**

SQLite 三张表：`documents`、`chunks`、`versions`。`DocumentMeta` 的 JSON 字段（如缺失）存 `NULL`。`upsert` 用 `INSERT ... ON CONFLICT DO UPDATE`。`get_chunks` 保持传入 id 的顺序返回。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/index/test_store.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/index/store.py tests/index/test_store.py
git commit -m "feat: SQLite metadata and chunk store

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 12: 稀疏索引（jieba + BM25）

**Files:**
- Create: `src/parking_kb/index/sparse.py`, `corpus/jieba_userdict.txt`, `tests/index/test_sparse.py`
- Test: `tests/index/test_sparse.py`

**Interfaces:**
- Consumes: `Chunk`
- Produces:
  - `tokenize_zh(text: str) -> list[str]`
  - `SparseIndex.build(chunks: list[Chunk]) -> SparseIndex`
  - `.search(query: str, top_k: int = 20) -> list[tuple[str, float]]`
  - `.save(path: Path) -> None`
  - `SparseIndex.load(path: Path) -> SparseIndex`

- [ ] **Step 1: 写失败测试**

```python
# tests/index/test_sparse.py
def test_bm25_ranks_relevant_chunk_first():
    idx = SparseIndex.build([
        chunk("c1", "d1", "停车收费按车型和停放时长计算"),
        chunk("c2", "d2", "道闸设备每月检查一次"),
    ])
    hits = idx.search("停车收费标准", top_k=2)
    assert hits[0][0] == "c1"

def test_sparse_index_persistence(tmp_path):
    idx = SparseIndex.build([chunk("c1", "d1", "月租车办理流程")])
    idx.save(tmp_path / "bm25.pkl")
    assert SparseIndex.load(tmp_path / "bm25.pkl").search("月租", top_k=1)[0][0] == "c1"
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/index/test_sparse.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`jieba.lcut` 后去掉空白、标点与停用词；保留数字与“收费/月租/车位/道闸/车牌”等停车术语。创建 `corpus/jieba_userdict.txt`（每行“词 词频 词性”，至少含上述术语），在 `tokenize_zh` 首次调用时加载。`BM25Okapi` 建索引；`save` 用 pickle 保存 `{chunk_ids, tokenized, k1, b}`；`load` 重建 BM25。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/index/test_sparse.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/index/sparse.py corpus/jieba_userdict.txt tests/index/test_sparse.py
git commit -m "feat: Chinese BM25 sparse index with persistence

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 13: 稠密索引（bge-m3 + Qdrant 本地模式）

**Files:**
- Create: `src/parking_kb/index/dense.py`, `tests/index/test_dense.py`
- Test: `tests/index/test_dense.py`

**Interfaces:**
- Consumes: `Chunk`
- Produces:
  - `Embedder` Protocol：`encode(texts: list[str]) -> list[list[float]]`
  - `BgeM3Embedder(model_name: str = "BAAI/bge-m3", device: str = "cuda")`
  - `DenseIndex(path: Path)`
  - `.build(chunks: list[Chunk], embedder: Embedder) -> None`
  - `.search(query: str, embedder: Embedder, top_k: int = 20) -> list[tuple[str, float]]`
  - `.close() -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/index/test_dense.py
class KeywordEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] if "收费" in t else [0.0, 1.0] for t in texts]

def test_dense_index_roundtrip(tmp_path):
    idx = DenseIndex(tmp_path / "q")
    idx.build([chunk("c1", "d1", "收费按小时计"), chunk("c2", "d2", "道闸维护")], KeywordEmbedder())
    idx.close()
    idx2 = DenseIndex(tmp_path / "q")
    hits = idx2.search("收费", KeywordEmbedder(), top_k=1)
    assert hits[0][0] == "c1"
    idx2.close()
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/index/test_dense.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`BgeM3Embedder` 懒加载 `SentenceTransformer`，`encode` 时 `normalize_embeddings=True`，批大小 32。`DenseIndex` 用 `QdrantClient(path=str(path))`；collection 名 `chunks`，距离 `COSINE`；payload 存 `doc_id`。`build` 先删旧 collection 再建。测试用固定假向量，不加载真实模型。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/index/test_dense.py -v`
Expected: PASS。

- [ ] **Step 5: 真实模型冒烟（可跳过）**

Run: `pytest tests/index/test_dense.py -v -m slow`
Expected: 若模型已缓存且 GPU 可用，能编码并返回结果；否则 skip。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/index/dense.py tests/index/test_dense.py
git commit -m "feat: bge-m3 dense index on Qdrant local mode

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 14: RRF 融合与混合检索

**Files:**
- Create: `src/parking_kb/retrieve/hybrid.py`, `tests/retrieve/test_hybrid.py`
- Test: `tests/retrieve/test_hybrid.py`

**Interfaces:**
- Consumes: `DenseIndex`、`SparseIndex`、`ChunkStore`、`Candidate`
- Produces:
  - `RetrievalConfig(dense_top_k=20, sparse_top_k=20, fused_top_k=20, rerank_top_k=8, rrf_k=60, use_dense=True, use_sparse=True, use_rerank=True, use_status_filter=True)`
  - `rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]`
  - `HybridRetriever(dense, sparse, store, embedder, reranker)`
  - `.retrieve(query: str, config: RetrievalConfig = RetrievalConfig()) -> list[Candidate]`

- [ ] **Step 1: 写失败测试**

```python
# tests/retrieve/test_hybrid.py
def test_rrf_prefers_chunk_in_both_lists():
    scores = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]

def test_dense_only_config_skips_sparse():
    retriever = make_retriever(dense_hits=["c1"], sparse_hits=["c2"])
    got = retriever.retrieve("q", RetrievalConfig(use_sparse=False, use_rerank=False, use_status_filter=False))
    assert got[0].chunk.chunk_id == "c1"
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/retrieve/test_hybrid.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`rrf_fuse` 对每个列表的每个 id 累加 `1/(k+rank)`。`retrieve`：按 config 用注入的 `embedder` 编码查询并调稠密索引，同时调稀疏索引；融合后取 `fused_top_k`，再经 `apply_status_filter`（Task 15）与注入的 `reranker` 重排（Task 16）；`Candidate.dense_rank/sparse_rank` 记录原始名次，供评估与调试。依赖通过构造函数注入，测试全部使用 Fake。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/retrieve/test_hybrid.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/retrieve/hybrid.py tests/retrieve/test_hybrid.py
git commit -m "feat: hybrid retrieval with RRF fusion

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 15: 时效/版本过滤

**Files:**
- Create: `src/parking_kb/retrieve/filters.py`, `tests/retrieve/test_filters.py`
- Test: `tests/retrieve/test_filters.py`

**Interfaces:**
- Consumes: `Candidate`、`ChunkStore`
- Produces:
  - `has_temporal_cue(query: str) -> bool`
  - `apply_status_filter(candidates: list[Candidate], query: str, store: ChunkStore) -> list[Candidate]`

- [ ] **Step 1: 写失败测试**

```python
# tests/retrieve/test_filters.py
def test_active_and_unknown_kept_inactive_excluded():
    cands = [cand("c1", status="现行"), cand("c2", status="失效"), cand("c3", status="未知")]
    got = apply_status_filter(cands, "停车怎么收费", store)
    assert [c.chunk.chunk_id for c in got] == ["c1", "c3"]

def test_unknown_is_deprioritized():
    got = apply_status_filter([cand("c1", status="未知"), cand("c2", status="现行")], "收费", store)
    assert got[0].chunk.chunk_id == "c2"

def test_temporal_cue_allows_old_version():
    got = apply_status_filter([cand("old", status="失效"), cand("new", status="现行")], "2023年的旧标准怎么收费", store)
    assert {c.chunk.chunk_id for c in got} == {"old", "new"}
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/retrieve/test_filters.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`has_temporal_cue` 匹配“旧/原/以前/历史/20XX 年/废止/之前”。默认剔除 `status == "失效"`；`未知` 保留但 `score *= 0.95`；同 `doc_no` 多版本时只保留 `status=="现行"` 中 `effective_date` 最大者（无时间线索时）。有线索时全部保留，按原分数排序。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/retrieve/test_filters.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/retrieve/filters.py tests/retrieve/test_filters.py
git commit -m "feat: enforce effective-date and version filtering

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 16: 重排

**Files:**
- Create: `src/parking_kb/retrieve/rerank.py`, `tests/retrieve/test_rerank.py`
- Test: `tests/retrieve/test_rerank.py`

**Interfaces:**
- Consumes: `Candidate`
- Produces:
  - `Reranker` Protocol：`score(query: str, texts: list[str]) -> list[float]`
  - `BgeReranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cuda")`
  - `rerank(candidates: list[Candidate], query: str, reranker: Reranker, top_k: int = 8) -> list[Candidate]`

- [ ] **Step 1: 写失败测试**

```python
# tests/retrieve/test_rerank.py
class FakeReranker:
    def score(self, query, texts):
        return [10.0 if "收费标准" in t else 0.0 for t in texts]

def test_rerank_reorders_and_truncates():
    cands = [cand("c1", text="设备维护"), cand("c2", text="停车收费标准")]
    got = rerank(cands, "收费", FakeReranker(), top_k=1)
    assert [c.chunk.chunk_id for c in got] == ["c2"]

def test_rerank_empty_input():
    assert rerank([], "q", FakeReranker(), top_k=8) == []
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/retrieve/test_rerank.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`BgeReranker` 用 `sentence_transformers.CrossEncoder`，懒加载，`predict` 返回分数。`rerank` 把分数写回 `Candidate.score`，按降序取 `top_k`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/retrieve/test_rerank.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/retrieve/rerank.py tests/retrieve/test_rerank.py
git commit -m "feat: cross-encoder reranking for retrieved candidates

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 17: LLM 客户端、Prompt 与带引用生成

**Files:**
- Create: `src/parking_kb/generate/client.py`, `src/parking_kb/generate/prompt.py`, `src/parking_kb/generate/schema.py`, `prompts/answer_v1.md`, `tests/generate/test_generate.py`
- Test: `tests/generate/test_generate.py`

**Interfaces:**
- Consumes: `Candidate`、`AnswerResult`、`Citation`
- Produces:
  - `LLMProvider` Protocol：`complete(messages: list[dict[str, str]], temperature: float = 0.0) -> str`
  - `OpenAICompatibleProvider(base_url: str, api_key: str, model: str)`
  - `CachedProvider(inner: LLMProvider, cache_dir: Path)`
  - `build_context(chunks: list[Candidate]) -> str`
  - `parse_answer_json(raw: str, chunks: list[Candidate]) -> AnswerResult`
  - `AnswerGenerator(provider: LLMProvider, prompt_path: Path = Path("prompts/answer_v1.md"))`
  - `.generate(question: str, chunks: list[Candidate]) -> AnswerResult`

- [ ] **Step 1: 写失败测试**

```python
# tests/generate/test_generate.py
def test_generate_maps_numbered_citations_to_chunk_ids():
    llm = FakeLLM('{"answer":"按小时收费[1]","citations":[1],"abstained":false,"missing":[]}')
    got = AnswerGenerator(llm).generate("怎么收费", [cand("c1", text="按小时收费")])
    assert got.answer == "按小时收费[1]"
    assert got.citations[0].chunk_id == "c1"
    assert got.abstained is False

def test_generate_abstains_when_model_says_so():
    llm = FakeLLM('{"answer":"资料中未提及","citations":[],"abstained":true,"missing":["具体费率"]}')
    got = AnswerGenerator(llm).generate("明年费率", [cand("c1")])
    assert got.abstained is True
    assert got.missing == ["具体费率"]

def test_context_includes_title_effective_date_and_page():
    ctx = build_context([cand("c1", title="收费办法", effective_date="2026-01-01", page=3)])
    assert "收费办法" in ctx and "2026-01-01" in ctx and "p.3" in ctx
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/generate/test_generate.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

`OpenAICompatibleProvider` 用 `openai.OpenAI(base_url, api_key)`，`chat.completions.create`，`temperature=0`。`CachedProvider` 以 `sha256(model + json(messages) + temperature)` 为文件名，命中直接读。`prompts/answer_v1.md` 写明：只用上下文、逐句引用 `[n]`、证据不足拒答、版本冲突优先现行并说明生效日期、数字逐字引用。`parse_answer_json` 解析失败时重试一次，仍失败则返回 `abstained=True` 并记录原始输出。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/generate/test_generate.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/generate prompts/answer_v1.md tests/generate/test_generate.py
git commit -m "feat: cited answer generation with abstention and LLM cache

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 18: 评估指标

**Files:**
- Create: `src/parking_kb/eval/schema.py`, `src/parking_kb/eval/metrics.py`, `tests/eval/test_metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `GoldSpan(doc_id: str, char_start: int, char_end: int, quote: str)`
  - `EvalQuestion(qid, question, gold_answer, gold_span, question_type, difficulty, source_docs, verified=False)`
  - `recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float`
  - `mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float`
  - `ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float`
  - `citation_precision(predicted: list[str], gold: set[str]) -> float`
  - `citation_recall(predicted: list[str], gold: set[str]) -> float`
  - `abstention_accuracy(predicted: list[bool], gold: list[bool]) -> float`
  - `bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]`

- [ ] **Step 1: 写失败测试**

```python
# tests/eval/test_metrics.py
def test_recall_and_mrr_hand_computed():
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert mrr_at_k(["a", "b"], {"b"}, 10) == 0.5

def test_ndcg_hand_computed():
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    assert round(ndcg_at_k(["a", "b"], {"b"}, 2), 4) == round(1 / math.log2(3), 4)

def test_bootstrap_ci_is_deterministic():
    assert bootstrap_ci([1.0, 0.0, 1.0], seed=0) == bootstrap_ci([1.0, 0.0, 1.0], seed=0)
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

按标准定义实现；`ndcg_at_k` 使用 `1/log2(rank+2)` 折扣，单相关文档时按二值增益。`bootstrap_ci` 用 `random.Random(seed)` 有放回抽样，返回 2.5%/97.5% 分位。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/eval/schema.py src/parking_kb/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: retrieval and citation evaluation metrics

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 19: 评估集构建与分层抽样

**Files:**
- Create: `src/parking_kb/eval/build_set.py`, `scripts/build_eval_set.py`, `tests/eval/test_build_set.py`
- Test: `tests/eval/test_build_set.py`

**Interfaces:**
- Consumes: `Chunk`、`LLMProvider`、`EvalQuestion`、`GoldSpan`
- Produces:
  - `QUESTION_TYPES = ("fact", "table", "cross_doc", "version", "unanswerable")`
  - `build_eval_set(chunks: list[Chunk], llm: LLMProvider, per_type: dict[str, int]) -> list[EvalQuestion]`
  - `write_eval_set(questions: list[EvalQuestion], path: Path) -> None`
  - `load_eval_set(path: Path) -> list[EvalQuestion]`
  - `resolve_gold_chunks(question: EvalQuestion, chunks: list[Chunk]) -> list[str]`（按 `gold_span` 的字符区间映射）

- [ ] **Step 1: 写失败测试**

```python
# tests/eval/test_build_set.py
def test_build_eval_set_returns_valid_questions():
    llm = FakeJsonLLM({'questions': [{'question': '收费标准是什么', 'answer': '按小时', 'quote': '按小时收费', 'type': 'fact', 'difficulty': 'easy'}]})
    qs = build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"fact": 1})
    assert len(qs) == 1
    assert qs[0].question == "收费标准是什么"
    assert qs[0].gold_span.quote == "按小时收费"
    assert qs[0].verified is False

def test_resolve_gold_chunks_maps_span():
    q = question_with_span("c1", quote="按小时收费", char_start=0, char_end=5)
    assert resolve_gold_chunks(q, [chunk("c1", "d1", "按小时收费。")]) == ["c1"]

def test_build_eval_set_skips_unresolvable_gold():
    llm = FakeJsonLLM({'questions': [{'question': 'x', 'answer': 'y', 'quote': '不存在的句子', 'type': 'fact', 'difficulty': 'easy'}]})
    assert build_eval_set([chunk("c1", "d1", "按小时收费。")], llm, {"fact": 1}) == []
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/eval/test_build_set.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

按题型抽样 chunk（`table` 只选 table 块；`cross_doc` 选两个不同 doc；`version` 优先同 `doc_no` 多版本；`unanswerable` 由 LLM 生成语料外问题）。每类一个 prompt 模板，要求输出 JSON。`gold_span` 用原文 `quote` 在 chunk 中的 `str.find` 定位；找不到则丢弃该题并记日志。`unanswerable` 题使用空 `GoldSpan(doc_id="", char_start=0, char_end=0, quote="")`，`resolve_gold_chunks` 对这类题返回 `[]`，检索指标不纳入，只统计拒答准确率。`write_eval_set` 输出 JSONL；`verified` 默认 false。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/eval/test_build_set.py -v`
Expected: PASS。

- [ ] **Step 5: 生成候选评估集**

Run: `python scripts/build_eval_set.py --chunks data/chunks.jsonl --out eval/questions.jsonl --per-type fact=20,table=12,cross_doc=10,version=10,unanswerable=10`
Expected: 60~80 题；随后人工校对首批 20 题，把 `verified` 改为 true，并修正错误答案/出处。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/eval/build_set.py scripts/build_eval_set.py tests/eval/test_build_set.py eval/questions.jsonl
git commit -m "feat: build stratified evaluation set with span-anchored gold

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 20: LLM Judge（异源模型）

**Files:**
- Create: `src/parking_kb/eval/judge.py`, `tests/eval/test_judge.py`
- Test: `tests/eval/test_judge.py`

**Interfaces:**
- Consumes: `EvalQuestion`、`AnswerResult`、`Chunk`、`LLMProvider`
- Produces:
  - `Verdict(correct: Literal["correct","partial","wrong"], faithful: bool, citation_precise: bool, reason: str)`
  - `Judge` Protocol：`judge(question: EvalQuestion, answer: AnswerResult, contexts: list[Chunk]) -> Verdict`
  - `LLMJudge(provider: LLMProvider)`
  - `load_verdicts(path: Path) -> dict[str, Verdict]`
  - `save_verdicts(verdicts: dict[str, Verdict], path: Path) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/eval/test_judge.py
def test_judge_parses_verdict_json():
    llm = FakeLLM('{"correct":"correct","faithful":true,"citation_precise":true,"reason":"答案与出处一致"}')
    v = LLMJudge(llm).judge(question(), answer("按小时收费[1]"), [chunk("c1")])
    assert v.correct == "correct" and v.faithful is True

def test_judge_marks_invalid_json_as_wrong():
    v = LLMJudge(FakeLLM("not json")).judge(question(), answer("x"), [])
    assert v.correct == "wrong"
    assert "解析失败" in v.reason
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/eval/test_judge.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

Judge prompt 分两段：正确性对照 `gold_answer` 给三档；faithfulness 对答案逐句检查是否被 `contexts` 支持。要求输出 JSON。解析失败时返回 `Verdict("wrong", False, False, "解析失败: ...")`。`load/save_verdicts` 用 JSONL，key 为 `f"{qid}|{config}"`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/eval/test_judge.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/parking_kb/eval/judge.py tests/eval/test_judge.py
git commit -m "feat: independent LLM judge for answer and faithfulness

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 21: 评估运行器与消融报告

**Files:**
- Create: `src/parking_kb/eval/run.py`, `src/parking_kb/eval/report.py`, `scripts/eval.py`, `tests/eval/test_run.py`
- Test: `tests/eval/test_run.py`

**Interfaces:**
- Consumes: Task 14~20 的全部接口
- Produces:
  - `CONFIGS: dict[str, RetrievalConfig]`
  - `run_eval(questions: list[EvalQuestion], config_name: str, retriever, generator, judge, chunks: list[Chunk]) -> dict`
  - `aggregate(results: list[dict]) -> dict`
  - `render_eval_report(aggregated: dict, verdicts: list[dict]) -> str`
  - `write_judge_verdicts(verdicts: list[dict], path: Path) -> None`
  - `load_judge_verdicts(path: Path) -> list[dict]`

- [ ] **Step 1: 写失败测试**

```python
# tests/eval/test_run.py
def test_run_eval_produces_metrics_and_failures():
    result = run_eval([question()], "dense_only", fake_retriever(), fake_generator(), fake_judge(), [chunk("c1")])
    assert 0.0 <= result["recall@5"] <= 1.0
    assert "error_types" in result

def test_report_contains_ablation_table():
    text = render_eval_report({"dense_only": {"recall@5": 0.5}, "full": {"recall@5": 0.9}}, [])
    assert "dense_only" in text and "full" in text and "Recall@5" in text
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/eval/test_run.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

四个配置：`dense_only`（`use_sparse=False,use_rerank=False,use_status_filter=False`）、`dense_bm25_rrf`（`use_rerank=False,use_status_filter=False`）、`dense_bm25_rrf_rerank`（`use_status_filter=False`）、`full`。每题记录检索结果、生成结果、judge 判定、耗时与错误类型。`aggregate` 计算均值与 bootstrap 置信区间，分“已校对 / 未校对 / 全量”三组。`render_eval_report` 输出消融表、分题型表、置信区间、失败案例（每题最多 3 条）。判定结果写入 `eval/judge_verdicts.jsonl`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/eval/test_run.py -v`
Expected: PASS。

- [ ] **Step 5: 跑首次完整评估**

Run: `python scripts/eval.py --questions eval/questions.jsonl --configs dense_only,dense_bm25_rrf,dense_bm25_rrf_rerank,full --out reports/eval_report.md`
Expected: 报告含四组配置对比；已校对子集给出置信区间；失败案例分类。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/eval/run.py src/parking_kb/eval/report.py scripts/eval.py tests/eval/test_run.py reports/eval_report.md eval/judge_verdicts.jsonl
git commit -m "feat: ablation evaluation runner and report

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 22: API 与演示前端

**Files:**
- Create: `src/parking_kb/api/main.py`, `src/parking_kb/api/static/index.html`, `app.js`, `styles.css`, `tests/api/test_api.py`
- Test: `tests/api/test_api.py`

**Interfaces:**
- Consumes: `HybridRetriever`、`AnswerGenerator`、`ChunkStore`
- Produces:
  - `ChatRequest(question: str)`
  - `ChatResponse(answer: str, citations: list[dict], abstained: bool, latency_ms: int)`
  - `ChatService(retriever, generator, store)`
  - `.chat(question: str) -> ChatResponse`
  - `create_app(service: ChatService) -> FastAPI`
  - 路由：`GET /health`、`GET /documents`、`POST /chat`

- [ ] **Step 1: 写失败测试**

```python
# tests/api/test_api.py
def test_health():
    client = TestClient(create_app(fake_service()))
    assert client.get("/health").json() == {"status": "ok"}

def test_chat_returns_answer_and_citations():
    resp = TestClient(create_app(fake_service())).post("/chat", json={"question": "怎么收费"})
    body = resp.json()
    assert body["answer"]
    assert body["citations"][0]["chunk_id"] == "c1"

def test_empty_question_rejected():
    resp = TestClient(create_app(fake_service())).post("/chat", json={"question": "  "})
    assert resp.status_code == 422
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/api/test_api.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 API 与页面**

`/chat` 校验非空，调用 service，返回 JSON。`/documents` 返回 `store.list_documents()`。静态页面：输入框 + 消息列表；答案下方渲染引用编号，点击展开 `title / effective_date / section_path / page` 与原文片段；拒答时显示“资料中未提及”样式。`/health` 返回 `{"status":"ok"}`。

- [ ] **Step 4: 运行并确认通过**

Run: `pytest tests/api/test_api.py -v`
Expected: PASS。

- [ ] **Step 5: 手动验收**

Run: `uvicorn parking_kb.api.main:app --reload`
Expected: 浏览器提 10 个真实业务问题，答案带可点引用；扫描件、价目表、版本陷阱各至少 2 题。

- [ ] **Step 6: 提交**

```bash
git add src/parking_kb/api tests/api
git commit -m "feat: chat API and citation-aware demo UI

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## Task 23: 端到端集成、文档与复现脚本

**Files:**
- Create: `tests/integration/test_end_to_end.py`, `README.md`, `docs/decisions.md`, `docs/eval-methodology.md`, `Makefile`
- Test: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: 全部模块
- Produces: 可复现的演示与文档；`make ingest`、`make eval-report`、`make test`

- [ ] **Step 1: 写失败测试**

```python
# tests/integration/test_end_to_end.py
def test_ingest_retrieve_answer_with_fake_models(tmp_path):
    corpus, data = make_fixture_corpus(tmp_path)
    report = run_ingest(corpus, data)
    store = ChunkStore(data / "kb.sqlite"); store.upsert_chunks(load_chunks_jsonl(data / "chunks.jsonl"))
    retriever = make_retriever(store, data)
    got = retriever.retrieve("停车怎么收费", RetrievalConfig(use_rerank=False, use_status_filter=False))
    assert got and got[0].chunk.chunk_id
    ans = AnswerGenerator(FakeLLM(...)).generate("停车怎么收费", got[:3])
    assert ans.citations
```

- [ ] **Step 2: 运行并确认失败**

Run: `pytest tests/integration/test_end_to_end.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现集成测试与文档**

`README.md` 必须含：项目一句话介绍、Mermaid 架构图、快速开始（三条命令）、评估结果摘要表（实测数字，不用目标值冒充）、**逐项对照设计文档 §9 的验收表（达不到的写原因，不粉饰）**、设计决策摘要、局限与后续工作。`docs/decisions.md` 记录轻量 ADR（为什么 Qdrant 本地、为什么字符区间锚定、为什么未知状态降权、为什么 judge 异源）。`docs/eval-methodology.md` 说明题型、指标定义、置信区间与人工校对范围。`Makefile` 提供 `ingest`、`eval-report`、`test`、`dev`。

- [ ] **Step 4: 运行并确认通过**

Run: `make test`
Expected: 全部单元与集成测试 PASS，不联网。

- [ ] **Step 5: 验证离线复现**

Run: `make eval-report`
Expected: 不联网、无需 API key，读取 `eval/judge_verdicts.jsonl` 重算并重写 `reports/eval_report.md`，指标与提交版本一致。

- [ ] **Step 6: 提交**

```bash
git add tests/integration README.md docs/decisions.md docs/eval-methodology.md Makefile
git commit -m "docs: end-to-end integration, reproducibility, and project README

Spec: 2026-10-26-parking-kb-rag-design.md"
```

---

## 附录 A：种子语料清单

以下链接已核实可公开访问；下载时在 `corpus/sources.yaml` 记录来源、类型与许可说明，并由脚本生成 `corpus/sources.md`。

**政策法规 / 标准**
1. DB4404/T 66—2024《停车场运营管理规范》— https://dbba.sacinfo.org.cn/attachment/downloadStdFile?pk=78589883f7c6cd4a680b8d8388587187b6a199e2faf2e9e5c5ff7d62ca94cade
2. 广东省住建厅智慧停车技术导则 — http://zfcxjst.gd.gov.cn/attachment/0/605/605578/4851723.pdf
3. T/CBDSA《公共停车场（库）运营服务规范》— https://www.ttbz.org.cn/upload/file/20250917/6389371666512928364239674.pdf
4. T/DGPIA 001—2021《停车场运营管理服务规范》— https://www.ttbz.org.cn/upload/file/20210824/6376539911318220821196004.pdf
5. 长沙市《重点区域公共停车场管理规范》— http://amr.changsha.gov.cn/zfxxgk/tzgg_1_1/202507/P020250709603698624754.docx
6. 《机动车停放服务收费管理办法》（2000，已失效，用于时效陷阱）— https://www.gov.cn/gongbao/content/2000/content_60437.htm

**收费标准**
7. 湖南省机动车停放服务收费管理办法（湘发改价费规〔2025〕723号）— https://fgw.changde.gov.cn/upload/fgw/application/2026/01/09/%e5%85%b3%e4%ba%8e%e5%8d%b0%e5%8f%91%e3%80%8a%e6%b9%96%e5%8d%97%e7%9c%81%e6%9c%ba%e5%8a%a8%e8%bd%a6%e5%81%9c%e6%94%be%e6%9c%8d%e5%8a%a1%e6%94%b6%e8%b4%b9%e7%ae%a1%e7%90%86%e5%8a%9e%e6%b3%95%e3%80%8b%e7%9a%84%e9%80%9a%e7%9f%a5(%e6%b9%98%e5%8f%91%e6%94%b9%e4%bb%b7%e8%b4%b9%e8%a7%84%e3%80%942025%e3%80%95723%e5%8f%b7).pdf
8. 湖北省机动车停放服务收费管理办法 — http://fgw.hubei.gov.cn/material/pdf/web/viewer.html?file=//fgw.hubei.gov.cn/fgjj/ztzl/zl/2023/jgfw/zc_78952/tcsf/202311/P020240222671923030465.pdf
9. 安徽省机动车停放服务收费管理办法 — https://www.cnjx.gov.cn/upload_xc/download?siteId=7&id=1102966

**设备手册**
10. ZKTeco 熵基下载中心 — https://www.zkinte.com/category/downloads/manuals/page/11
11. 太原飞凡科技下载中心 — https://www.flyfan.net/h-col-117.html
12. 大华 Parking Management System 手册 — https://files.dahua.support/Instrukcje%20obs%c5%82ugi/Angielskie/PMS_Users+Manual_V1.03.1.pdf
13. 云盟车牌识别一体机调试指南 — https://ymzhdoc.szymzh.com/doc/1282/

**运营 SOP / 客服话术**
14. 《道路内智能泊车客服中心服务规范》团体标准 — https://www.ttbz.org.cn/upload/file/20220621/6379142643651665114606201.pdf
15. 扬州停车客服培训（4W1H、3F 法则）— http://www.chinaparking.org/newsid_33686
16. 临停超时现场话术 — http://mp.weixin.qq.com/s?__biz=Mzg2NDg3OTUzOA==&mid=2247485324&idx=1&sn=78170a75b712b1f25f0af2a947a1e2df
17. 江苏无人值守坐席平台采购文档 — http://www.ccgp-jiangsu.gov.cn/fileApi/320000/1a2a74f7892640679204245ff15c0901.doc

## 附录 B：环境准备

```bash
cd "D:/VS工作/parking-kb-rag"
python -m venv .venv
source .venv/Scripts/activate
pip install -U pip
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available())"
```

若 `torch.cuda.is_available()` 为 False，重装匹配 CUDA 的 torch；仍不可用则把 `Settings.device` 设为 `cpu`，并在 README 标注性能差异。若 `rapidocr-onnxruntime` 安装失败，改用 `rapidocr` 包，并保持 `OcrEngine` 协议不变。

`.env.example`：

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
DEVICE=cuda
```
