from cua.safety.redaction import redact_detail, redact_text


def test_redacts_api_key_like_value():
    out = redact_text("Authorization: Bearer abcdef1234567890")
    assert "abcdef1234567890" not in out


def test_redacts_sk_prefixed_token():
    out = redact_text("token=sk-abcdef1234567890xyz")
    assert "sk-abcdef1234567890xyz" not in out


def test_redacts_card_number_shaped_digits():
    out = redact_text("card 4111111111111111 on file")
    assert "4111111111111111" not in out


def test_redacts_ssn_shaped_value():
    out = redact_text("ssn 123-45-6789")
    assert "123-45-6789" not in out


def test_ordinary_text_is_untouched():
    assert redact_text("member 12345 has balance 200") == "member 12345 has balance 200"


def test_redact_detail_recurses_into_nested_dicts():
    detail = {"outer": {"inner": "bearer abcdefghijklmno"}}
    out = redact_detail(detail)
    assert "abcdefghijklmno" not in out["outer"]["inner"]


def test_redact_detail_leaves_non_strings_untouched():
    detail = {"count": 3, "ok": True, "nothing": None}
    assert redact_detail(detail) == detail
