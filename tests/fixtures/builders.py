from pathlib import Path

import pymupdf as fitz


def make_text_pdf(
    path: Path,
    text: str = (
        "停车收费标准按照车型和停放时长计算，小型车每小时收费两元。\n"
        "大型车每小时收费五元，夜间停放按次收费，不足一小时按一小时计。"
    ),
) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    for i, line in enumerate(text.split("\n")):
        page.insert_text((72, 72 + i * 18), line, fontname="china-s", fontsize=12)
    doc.save(path)
    doc.close()
    return path


def make_image_only_pdf(path: Path) -> Path:
    """A page with no text layer, used to exercise the scanned-PDF path."""
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(page.rect, color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(path)
    doc.close()
    return path


def make_mixed_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "第一章 总则", fontname="china-s", fontsize=20)
    page.insert_text((72, 110), "停车收费按小时计算。", fontname="china-s", fontsize=10)
    x0, y0 = 72, 150
    rows = [["车型", "费率"], ["小型车", "2元/小时"]]
    cell_w, cell_h = 100, 30
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = fitz.Rect(
                x0 + c * cell_w,
                y0 + r * cell_h,
                x0 + (c + 1) * cell_w,
                y0 + (r + 1) * cell_h,
            )
            page.draw_rect(cell, color=(0, 0, 0), width=1)
            page.insert_textbox(
                cell + (4, 4, -4, -4), value, fontname="china-s", fontsize=10
            )
    doc.save(path)
    doc.close()
    return path


def make_docx_with_headings(path: Path) -> Path:
    from docx import Document

    doc = Document()
    doc.add_heading("第一章 总则", level=1)
    doc.add_paragraph("停车收费按小时计算。")
    doc.save(path)
    return path


def make_xlsx_with_merged_header(path: Path) -> Path:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "价目"
    ws.merge_cells("A1:B1")
    ws["A1"] = "车型"
    ws.merge_cells("A2:A3")
    ws["A2"] = "小型车"
    ws["B2"] = "2元/小时"
    ws["B3"] = "3元/小时"
    wb.save(path)
    return path


def make_fixture_corpus(root: Path) -> tuple[Path, Path]:
    corpus = root / "corpus"
    data = root / "data"
    corpus.mkdir()
    make_text_pdf(corpus / "ok.pdf")
    make_docx_with_headings(corpus / "ok.docx")
    (corpus / "bad.pdf").write_bytes(b"not a pdf")
    return corpus, data
