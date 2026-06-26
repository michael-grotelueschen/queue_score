"""Tests for ``build_features`` -- data cleaning, missing data, derived flags,
and the no-leakage guarantee. All use small synthetic frames (the ``make_raw``
fixture)."""

import numpy as np
import pandas as pd

import train


# --- structure -----------------------------------------------------------

def test_columns_and_index_preserved(make_raw):
    raw = make_raw([{}, {}, {}])
    feats = train.build_features(raw)
    assert list(feats.columns) == train.FEATURES
    assert feats.index.equals(raw.index)
    assert len(feats) == 3


# --- missing categorical data -------------------------------------------

def test_missing_categoricals_become_unknown(make_raw):
    raw = make_raw([{"region": np.nan, "state": np.nan,
                     "type_clean": np.nan, "service": np.nan}])
    feats = train.build_features(raw)
    for col in ["region", "state", "type_clean", "service"]:
        assert feats.loc[0, col] == "Unknown"


def test_all_missing_row_yields_complete_feature_row(make_raw):
    raw = make_raw([{c: np.nan for c in
                     ["region", "state", "type_clean", "service",
                      "mw1", "mw2", "q_year", "IA_status_clean", "ia_date"]}])
    feats = train.build_features(raw)  # must not raise
    # Categoricals + flags are never NaN; only numerics may be NaN.
    for col in ["region", "state", "type_clean", "service",
                "is_hybrid", "ia_executed"]:
        assert feats[col].notna().all()


# --- missing / invalid numeric (capacity) data --------------------------

def test_missing_mw_gives_nan_log_mw(make_raw):
    feats = train.build_features(make_raw([{"mw1": np.nan}]))
    assert pd.isna(feats.loc[0, "log_mw"])


def test_nonpositive_mw_treated_as_missing(make_raw):
    # The real data contains negative and zero mw1; never log10 a non-positive.
    raw = make_raw([{"mw1": -671.0}, {"mw1": 0.0}, {"mw1": 100.0}])
    feats = train.build_features(raw)
    assert pd.isna(feats.loc[0, "log_mw"])
    assert pd.isna(feats.loc[1, "log_mw"])
    assert feats.loc[2, "log_mw"] == np.log10(100.0)
    assert np.isfinite(feats["log_mw"]).sum() == 1


def test_log_mw_value(make_raw):
    feats = train.build_features(make_raw([{"mw1": 1000.0}]))
    assert feats.loc[0, "log_mw"] == 3.0


# --- queue year clipping -------------------------------------------------

def test_q_year_clipped_to_training_window(make_raw):
    lo, hi = train.TRAIN_YEAR_MIN, train.TRAIN_YEAR_MAX
    raw = make_raw([{"q_year": 1995}, {"q_year": 2023}, {"q_year": 2010}])
    feats = train.build_features(raw)
    assert feats.loc[0, "q_year_clipped"] == lo
    assert feats.loc[1, "q_year_clipped"] == hi
    assert feats.loc[2, "q_year_clipped"] == 2010


def test_missing_q_year_propagates_as_nan(make_raw):
    feats = train.build_features(make_raw([{"q_year": np.nan}]))
    assert pd.isna(feats.loc[0, "q_year_clipped"])


# --- derived flags -------------------------------------------------------

def test_is_hybrid_from_plus_in_type(make_raw):
    raw = make_raw([{"type_clean": "Solar+Battery", "mw2": np.nan},
                    {"type_clean": "Solar", "mw2": np.nan}])
    feats = train.build_features(raw)
    assert feats.loc[0, "is_hybrid"] == 1
    assert feats.loc[1, "is_hybrid"] == 0


def test_is_hybrid_from_second_capacity(make_raw):
    feats = train.build_features(make_raw([{"type_clean": "Solar", "mw2": 50.0}]))
    assert feats.loc[0, "is_hybrid"] == 1


def test_ia_executed_from_status(make_raw):
    rows = [{"IA_status_clean": s, "ia_date": np.nan}
            for s in train.IA_EXECUTED_STATUSES]
    rows.append({"IA_status_clean": "Feasibility Study", "ia_date": np.nan})
    feats = train.build_features(make_raw(rows))
    assert feats["ia_executed"].iloc[:-1].eq(1).all()
    assert feats["ia_executed"].iloc[-1] == 0


def test_ia_executed_from_ia_date(make_raw):
    feats = train.build_features(
        make_raw([{"IA_status_clean": "Feasibility Study", "ia_date": 43000.0}])
    )
    assert feats.loc[0, "ia_executed"] == 1


def test_flags_are_int_zero_one(make_raw):
    feats = train.build_features(make_raw([{}, {"type_clean": "Solar+Battery"}]))
    for col in ["is_hybrid", "ia_executed"]:
        assert set(feats[col].unique()) <= {0, 1}
        assert feats[col].dtype.kind == "i"


# --- leakage guard -------------------------------------------------------

def test_features_independent_of_outcome_columns(make_raw):
    """Mutating final-disposition columns must not change any feature."""
    base = make_raw([{"q_status": "withdrawn", "wd_date": 43000.0, "on_date": np.nan}])
    leaked = make_raw([{"q_status": "operational", "wd_date": np.nan, "on_date": 44000.0}])
    f_base = train.build_features(base)
    f_leaked = train.build_features(leaked)
    pd.testing.assert_frame_equal(f_base, f_leaked)


def test_ia_executed_does_not_use_disposition(make_raw):
    # Same IA milestone inputs, opposite outcomes -> identical ia_executed.
    withdrawn = make_raw([{"q_status": "withdrawn", "IA_status_clean": "IA Executed"}])
    operational = make_raw([{"q_status": "operational", "IA_status_clean": "IA Executed"}])
    assert (train.build_features(withdrawn).loc[0, "ia_executed"]
            == train.build_features(operational).loc[0, "ia_executed"] == 1)
