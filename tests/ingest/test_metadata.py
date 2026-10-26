from parking_kb.ingest.metadata import (
    extract_metadata,
    load_overrides,
    parse_doc_no,
    parse_effective_date,
)
from tests.factories import file_info, parsed_doc_with_text


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


def test_partial_override_keeps_regex_fields():
    doc = parsed_doc_with_text("湘发改价费规〔2025〕723号\n本办法自2026年1月1日起施行。")
    meta = extract_metadata(doc, file_info(), {file_info().sha256: {"status": "失效"}})
    assert meta.status == "失效"
    assert meta.doc_no == "湘发改价费规〔2025〕723号"
    assert meta.effective_date == "2026-01-01"


def test_unknown_status_when_no_signal():
    meta = extract_metadata(parsed_doc_with_text("普通通知"), file_info(), {})
    assert meta.status == "未知"


def test_parse_helpers_reject_malformed_values():
    assert parse_doc_no("没有文号") is None
    assert parse_effective_date("自2026年13月1日起施行") is None


def test_parse_helpers_tolerate_spacing_and_实施():
    assert parse_doc_no("鄂价工服〔2016〕151 号") == "鄂价工服〔2016〕151 号"
    assert parse_effective_date("本办法自2026 年1 月1 日起实施") == "2026-01-01"


def test_status_ignores_technical_failure_mentions():
    doc = parsed_doc_with_text("当感知设备失效时，系统应切换至热备服务。")
    assert extract_metadata(doc, file_info(), {}).status == "未知"


def test_status_ignores_repeal_of_other_documents():
    doc = parsed_doc_with_text(
        "本办法自2017年2月1日起施行。《旧办法》（鄂价工服规〔2013〕83号）同时废止。"
    )
    assert extract_metadata(doc, file_info(), {}).status == "现行"


def test_status_expired_when_self_repealed():
    doc = parsed_doc_with_text("本办法自2024年1月1日起废止。")
    assert extract_metadata(doc, file_info(), {}).status == "失效"


def test_load_overrides(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "overrides:\n  abc123:\n    status: 失效\n    version: '2016'\n",
        encoding="utf-8",
    )
    overrides = load_overrides(path)
    assert overrides["abc123"]["status"] == "失效"
    assert load_overrides(tmp_path / "missing.yaml") == {}
