import pytest

from src.timeutils import format_timestamp, parse_timestamp


@pytest.mark.parametrize("text,expected", [
    ("01:05.60", 65.6), ("00:00.00", 0.0),
    ("1:02:03.50", 3723.5), ("01:05", 65.0),
])
def test_parse_timestamp(text, expected):
    assert parse_timestamp(text) == expected


@pytest.mark.parametrize("text", ["bad", "-01:05.00", "00:60.00", "1:60:03.50", "NaN"])
def test_invalid_timestamp(text):
    with pytest.raises(ValueError):
        parse_timestamp(text)


def test_format_timestamp():
    assert format_timestamp(65.6) == "01:05.60"
    assert format_timestamp(59.999) == "01:00.00"
    assert format_timestamp(3723.5) == "62:03.50"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_format_value(value):
    with pytest.raises(ValueError):
        format_timestamp(value)
