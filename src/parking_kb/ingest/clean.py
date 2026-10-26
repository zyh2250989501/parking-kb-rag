import math
import re
from dataclasses import replace

from parking_kb.types import Block, ParsedDoc

SENTENCE_END = "。！？；：!?;:"
_LIST_START = re.compile(
    r"^(?:"
    r"第[一二三四五六七八九十百零\d]+[章节条款部分]"
    r"|[一二三四五六七八九十]+[、.]"
    r"|\d+[、.)）]"
    r"|[（(]\d+[)）]"
    r"|[（(][一二三四五六七八九十]+[)）]"
    r"|[-•·*]\s"
    r"|#"
    r")"
)
_PAGE_NUMBER = re.compile(
    r"^(?:"
    r"第\s*\d+\s*页(?:\s*[，,]?\s*共\s*\d+\s*页)?"
    r"|\d+"
    r"|[-—–]\s*\d+\s*[-—–]"
    r"|page\s*\d+(?:\s*of\s*\d+)?"
    r")$",
    re.IGNORECASE,
)
TEXT_KINDS = {"paragraph", "ocr_text"}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def join_wrapped_lines(text: str) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    joined = [lines[0]]
    for line in lines[1:]:
        previous = joined[-1]
        if previous and previous[-1] not in SENTENCE_END and not _LIST_START.match(line):
            ascii_word = (
                previous[-1].isascii()
                and previous[-1].isalnum()
                and line[0].isascii()
                and line[0].isalnum()
            )
            joined[-1] = previous + (" " if ascii_word else "") + line
        else:
            joined.append(line)
    return "\n".join(joined)


def _repeat_key(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if _PAGE_NUMBER.match(compact):
        return "第#页"
    return compact


def remove_repeated_lines(
    blocks: list[Block], page_count: int, threshold: float = 0.6
) -> list[Block]:
    if page_count < 2:
        return blocks
    needed = max(2, math.ceil(threshold * page_count))
    pages: dict[str, set[int]] = {}
    per_page: dict[tuple[str, int], int] = {}
    for block in blocks:
        if block.kind not in TEXT_KINDS or block.page is None:
            continue
        stripped = block.text.strip()
        if len(stripped) >= 40 or stripped[-1] in SENTENCE_END:
            continue
        key = _repeat_key(stripped)
        if not key:
            continue
        pages.setdefault(key, set()).add(block.page)
        per_page[(key, block.page)] = per_page.get((key, block.page), 0) + 1

    repeated = {
        key
        for key, seen_pages in pages.items()
        if len(seen_pages) >= needed
        and all(per_page[(key, page)] == 1 for page in seen_pages)
    }
    return [
        block
        for block in blocks
        if not (
            block.kind in TEXT_KINDS
            and block.page is not None
            and _repeat_key(block.text) in repeated
        )
    ]


def _grams(text: str, size: int = 5) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i : i + size] for i in range(len(compact) - size + 1)}


def _jaccard(left: str, right: str) -> float:
    left_grams, right_grams = _grams(left), _grams(right)
    if not left_grams or not right_grams:
        return 1.0 if left.strip() == right.strip() else 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def dedupe_blocks(blocks: list[Block], threshold: float = 0.9) -> list[Block]:
    kept: list[Block] = []
    seen_texts: list[str] = []
    for block in blocks:
        if block.kind in TEXT_KINDS:
            duplicate = any(_jaccard(block.text, text) > threshold for text in seen_texts)
            if duplicate:
                continue
            seen_texts.append(block.text)
        kept.append(block)
    return kept


def clean_document(doc: ParsedDoc) -> ParsedDoc:
    blocks: list[Block] = []
    for block in doc.blocks:
        text = normalize_text(block.text)
        if not text:
            continue
        if block.kind in TEXT_KINDS:
            text = join_wrapped_lines(text)
        elif block.kind == "heading":
            text = re.sub(r"\s*\n\s*", " ", text)
        blocks.append(replace(block, text=text))
    blocks = remove_repeated_lines(blocks, doc.page_count)
    blocks = dedupe_blocks(blocks)
    return replace(doc, blocks=blocks)
