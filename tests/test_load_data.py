"""Tests for ``load_data`` -- cleaning and row filtering against the real CSV.

These load the local LBNL snapshot and are skipped if it is not present, so the
suite still runs on a machine without the data file.
"""

import pytest

import train
from conftest import REAL_DATA

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason=f"LBNL workbook not available at {REAL_DATA}"
)

MODELED_STATUSES = {"active", "withdrawn", "operational", "suspended"}


@pytest.fixture(scope="module")
def loaded():
    return train.load_data()


def test_preamble_rows_skipped(loaded):
    # Real header becomes the columns; banner text is not a column or a value.
    assert {"q_id", "q_status", "q_date", "region"} <= set(loaded.columns)
    assert "RETURN TO CONTENTS" not in set(loaded.columns)
    assert not (loaded["q_id"].astype(str) == "RETURN TO CONTENTS").any()


def test_na_string_parsed_as_missing(loaded):
    # "NA" should be NaN, never the literal two-character string.
    assert not loaded.isin(["NA"]).to_numpy().any()
    # A column known to contain "NA" in the source is now numeric/nullable.
    assert loaded["prop_date"].isna().any()


def test_only_modeled_statuses_kept(loaded):
    present = set(loaded["q_status"].dropna().unique())
    assert present <= MODELED_STATUSES
    assert "unknown" not in present


def test_nonempty_and_has_active_projects(loaded):
    assert len(loaded) > 0
    assert (loaded["q_status"] == "active").sum() > 0


def test_row_count_equals_status_sum(loaded):
    assert len(loaded) == loaded["q_status"].isin(MODELED_STATUSES).sum()
