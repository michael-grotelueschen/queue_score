"""Tests for ``excel_date`` -- Excel serial date parsing and missing dates."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest


def test_nan_returns_none(train_module):
    assert train_module.excel_date(np.nan) is None
    assert train_module.excel_date(None) is None
    assert train_module.excel_date(pd.NaT) is None


def test_handles_real_datetimes(train_module):
    # The through-2025 workbook reads dates as datetimes, not serials.
    assert train_module.excel_date(pd.Timestamp("2019-02-15")) == "2019-02-15"
    assert train_module.excel_date(dt.date(2005, 5, 25)) == "2005-05-25"
    assert train_module.excel_date(dt.datetime(2020, 1, 2, 13, 30)) == "2020-01-02"


@pytest.mark.parametrize(
    "serial,expected",
    [
        (43511, "2019-02-15"),  # golden value from the 1899-12-30 epoch
        (38497, "2005-05-25"),
        (0, "1899-12-30"),      # the epoch itself
    ],
)
def test_known_serials(train_module, serial, expected):
    assert train_module.excel_date(serial) == expected


def test_accepts_float_and_numeric_string(train_module):
    # Source CSV carries serials as numbers; floats and numeric strings parse.
    assert train_module.excel_date(43511.0) == "2019-02-15"
    assert train_module.excel_date("43511") == "2019-02-15"


def test_garbage_returns_none_not_raises(train_module):
    assert train_module.excel_date("not-a-date") is None


def test_out_of_range_returns_none(train_module):
    # Serial large enough to overflow timedelta -> handled, not raised.
    assert train_module.excel_date(1e18) is None


def test_output_roundtrips_with_fromisoformat(train_module):
    # yrs_in_queue later parses this string back with date.fromisoformat.
    iso = train_module.excel_date(43511)
    assert dt.date.fromisoformat(iso) == dt.date(2019, 2, 15)
