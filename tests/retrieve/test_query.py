from parking_kb.retrieve.query import strip_document_prefix

TITLES = [
    "大华停车场管理系统用户手册 V2.0.0",
    "广东省智慧停车建设指南（草案）",
    "湖南省机动车停放服务收费管理办法（2025）",
    "停车场运营管理规范（DB4404/T 66—2024）",
]


def test_strips_leading_document_clause():
    """文档名出现在开头的限定语里时，它是过滤条件不是语义，检索用后半句。"""
    query = "大华停车场管理系统V2.0.0中，系统要求的处理器型号是什么？"
    assert strip_document_prefix(query, TITLES) == "系统要求的处理器型号是什么？"


def test_strips_clause_even_without_version_token():
    query = "广东省智慧停车建设指南（草案）里，对数据服务有什么要求？"
    assert strip_document_prefix(query, TITLES) == "对数据服务有什么要求？"


def test_keeps_query_when_document_name_is_the_subject():
    """没有分句边界就不动——问句可能就是围绕这份文档本身。"""
    query = "重点区域公共停车场管理规范适用于长沙哪些重点区域的公共停车场？"
    assert strip_document_prefix(query, TITLES) == query


def test_keeps_query_without_any_document_reference():
    query = "停车场夜间照明有什么要求？"
    assert strip_document_prefix(query, TITLES) == query


def test_keeps_query_when_leading_clause_is_too_short():
    """前半句去掉后剩不下东西，说明那不是限定语，不剥。"""
    query = "大华停车场管理系统V2.0.0？"
    assert strip_document_prefix(query, TITLES) == query


def test_ignores_incidental_short_overlap():
    """和标题偶然重叠几个字不算提到文档。"""
    query = "停车场里，车辆丢失了谁来赔？"
    assert strip_document_prefix(query, TITLES) == query
