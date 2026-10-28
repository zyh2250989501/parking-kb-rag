"""给英文文档生成中文检索描述（doc2query）。

英文手册的正文和中文查询之间没有词面交集，中文问「怎么添加操作员账号」
打不中 "User Management"。这里为每个英文 chunk 生成一段中文描述，
只用于建索引（`Chunk.search_text`），生成与引用展示仍用原文。
"""

import hashlib

from parking_kb.index.search_text import needs_translation
from parking_kb.types import Chunk

MAX_DESCRIPTION_CHARS = 300

SYSTEM_PROMPT = """你是中文知识库问答系统的检索增强模块。
用户会用中文提问，而原文是英文资料。请为给定段落写一段中文检索文本，让用户的中文问题能命中这段内容。

输出两行：
第一行：一句话说明这段讲的**功能、操作或对象**，不超过 60 字。
第二行：以「问：」开头，写 3~5 个用户可能提出的中文问题，用「；」分隔。要覆盖用户的实际说法与同义说法——中文里同一对象常有好几种叫法（如「用户」「操作员」「账号」「管理员」）。

要求：只能使用原文提供的信息，不得补充原文没有的内容，不得推测。不要分点、标题或其他解释。"""


def build_messages(chunk: Chunk) -> list[dict[str, str]]:
    user = (
        f"章节：{chunk.section_path}\n\n"
        f"原文：\n{chunk.text}\n\n"
        "请输出这段内容的中文检索文本。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def clean_description(raw: str) -> str:
    text = " ".join(raw.split()).strip().strip('"').strip("“”").strip()
    return text[:MAX_DESCRIPTION_CHARS]


def text_fingerprint(text: str) -> str:
    """描述是按文本内容生成的；chunk_id 只保证位置，重新入库后同位置的文本会变。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def description_is_current(chunk: Chunk, record: dict) -> bool:
    """sidecar 里这条描述是否还对得上当前文本。没有指纹的旧记录一律按过期处理。"""
    return bool(record.get("search_text")) and record.get("text_sha") == text_fingerprint(
        chunk.text
    )


def chunks_to_describe(chunks: list[Chunk]) -> list[Chunk]:
    """只处理英文且还没有描述的 chunk。"""
    return [
        chunk
        for chunk in chunks
        if not chunk.search_text and needs_translation(chunk.text)
    ]
