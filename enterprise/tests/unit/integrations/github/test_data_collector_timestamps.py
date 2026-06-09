from datetime import datetime

import pytest
from integrations.github.data_collector import _github_ts_to_naive_utc


def test_z_suffix_parsed_as_naive_utc():
    result = _github_ts_to_naive_utc('2025-06-19T21:19:36Z')
    # Must be naive (tzinfo stripped) so it can bind to a
    # TIMESTAMP WITHOUT TIME ZONE column via asyncpg.
    assert result == datetime(2025, 6, 19, 21, 19, 36)
    assert result.tzinfo is None


def test_non_utc_offset_converted_to_utc():
    # 21:19:36+02:00 == 19:19:36 UTC
    result = _github_ts_to_naive_utc('2025-06-19T21:19:36+02:00')
    assert result == datetime(2025, 6, 19, 19, 19, 36)
    assert result.tzinfo is None


@pytest.mark.parametrize('value', [None, ''])
def test_missing_values_return_none(value):
    assert _github_ts_to_naive_utc(value) is None
