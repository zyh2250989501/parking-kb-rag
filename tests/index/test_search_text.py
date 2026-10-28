from parking_kb.index.search_text import index_text, needs_translation
from tests.factories import chunk


def test_index_text_falls_back_to_original_text():
    assert index_text(chunk("c1", "d1", "收费按小时计")) == "收费按小时计"


def test_index_text_prefers_search_text():
    item = chunk(
        "c1",
        "d1",
        "You can add vehicle owner.",
        search_text="用户管理 添加操作员账号",
    )
    assert index_text(item) == "用户管理 添加操作员账号"


def test_needs_translation_detects_english_manual_text():
    assert needs_translation(
        "You can add vehicle owner, modify or delete the added owner."
    )


def test_needs_translation_skips_chinese_text():
    assert not needs_translation("停车场应配建或预留电动汽车充电基础设施。")


def test_needs_translation_skips_chinese_text_with_english_terms():
    """中文规范里的国标号与单位（lux、GB 50034）不应触发翻译。"""
    assert not needs_translation(
        "机动车库（车道）维持照度大于等于30lux，依据GB 50034-2013。"
    )
