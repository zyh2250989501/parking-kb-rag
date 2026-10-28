import re

from parking_kb.types import Chunk

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def index_text(chunk: Chunk) -> str:
    """索引该用哪段文本：有检索描述用描述，否则用原文。"""
    return chunk.search_text or chunk.text


def needs_translation(text: str) -> bool:
    """英文字母明显多于汉字时认为需要中文检索描述。

    中文规范里夹带的国标号与单位（GB 50034、lux）字母量很少，
    2 倍阈值把这类段落排除在外。
    """
    return len(_LATIN.findall(text)) > 2 * len(_CJK.findall(text))
