from parking_kb.index.describe import (
    build_messages,
    chunks_to_describe,
    clean_description,
    description_is_current,
    text_fingerprint,
)
from tests.factories import chunk


def test_build_messages_carries_section_and_text():
    messages = build_messages(
        chunk(
            "c1",
            "d1",
            "You can add vehicle owner.",
            section_path="3.7.2 Vehicle Owner Management",
        )
    )
    user = messages[-1]
    assert user["role"] == "user"
    assert "3.7.2 Vehicle Owner Management" in user["content"]
    assert "You can add vehicle owner." in user["content"]


def test_clean_description_strips_whitespace_and_quotes():
    assert clean_description('  "用户管理 添加账号" \n') == "用户管理 添加账号"


def test_clean_description_collapses_newlines():
    assert clean_description("用户管理\n添加账号") == "用户管理 添加账号"


def test_clean_description_drops_blank_response():
    assert clean_description("   ") == ""


def test_chunks_to_describe_skips_chinese_chunks():
    chunks = [
        chunk("c1", "d1", "停车场应配建或预留充电基础设施。"),
        chunk("c2", "d1", "You can add vehicle owner."),
    ]
    assert [c.chunk_id for c in chunks_to_describe(chunks)] == ["c2"]


def test_chunks_to_describe_skips_already_described():
    chunks = [
        chunk("c1", "d1", "You can add vehicle owner.", search_text="已有描述"),
        chunk("c2", "d1", "You can delete parking card."),
    ]
    assert [c.chunk_id for c in chunks_to_describe(chunks)] == ["c2"]


def test_description_is_current_when_text_is_unchanged():
    target = chunk("c1", "d1", "You can add vehicle owner.")
    record = {
        "search_text": "用户管理",
        "text_sha": text_fingerprint(target.text),
    }
    assert description_is_current(target, record)


def test_description_is_stale_when_text_changed():
    """chunk_id 只保证位置，重新入库后同一位置的文本可能换了。"""
    target = chunk("c1", "d1", "You can add vehicle owner.")
    record = {
        "search_text": "用户管理",
        "text_sha": text_fingerprint("You can add an operator account."),
    }
    assert not description_is_current(target, record)


def test_description_without_fingerprint_is_stale():
    """旧 sidecar 没有 text_sha，无法确认对应哪段文本，按过期处理。"""
    target = chunk("c1", "d1", "You can add vehicle owner.")
    assert not description_is_current(target, {"search_text": "用户管理"})
