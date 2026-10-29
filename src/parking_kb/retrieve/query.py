"""检索查询归一化。

问句里常带「文档名 + 中，……」这样的限定语，它是过滤条件，不是语义内容。
稠密检索会把限定语一起编码，把结果拉向提到这份文档的封面页——12 道表格题里
丢掉的 3 道，问句开头都是「大华停车场管理系统V2.0.0中，……」（ADR-011）。

这里只做一件事：认出开头的文档限定语，把它从**检索查询**里去掉。生成与引用
仍然用原问题——用户问的是哪份文档，答案里得说清楚。
"""

CLAUSE_BREAKS = "，,；;：:"

# 少于 6 字的重叠多是「停车场里」「系统里」这类普通词汇，不足以认成文档名。
MIN_TITLE_PREFIX = 6

# 剥完剩下太短，说明前半句本身就是问题，不是限定语。
MIN_REMAINDER = 4


def _title_prefix_length(clause: str, titles: list[str]) -> int:
    """clause 开头与任一标题开头的最长重合字数。"""
    best = 0
    for title in titles:
        limit = min(len(title), len(clause))
        length = 0
        while length < limit and title[length] == clause[length]:
            length += 1
        best = max(best, length)
    return best


def strip_document_prefix(query: str, titles: list[str]) -> str:
    """去掉问句开头的文档限定语，返回用作检索的查询。

    三个条件同时满足才剥，否则原样返回：

    - 句中存在分句边界（问句确实分成「限定语」和「问题」两段）
    - 限定语以某个标题的开头起手，允许省略尾部——「大华停车场管理系统V2.0.0」
      对得上「大华停车场管理系统用户手册 V2.0.0」
    - 去掉之后剩下的问题够长
    """
    cuts = [index for index, char in enumerate(query) if char in CLAUSE_BREAKS]
    if not cuts:
        return query
    cut = cuts[0]
    remainder = query[cut + 1 :].strip()
    if len(remainder) < MIN_REMAINDER:
        return query
    if _title_prefix_length(query[:cut], titles) < MIN_TITLE_PREFIX:
        return query
    return remainder
