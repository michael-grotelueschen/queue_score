"""Tests for the model constructors and the SHAP contribution helper.

Covers robustness to missing/unseen data in the preprocessing pipelines and the
exact-additivity contract of ``rf_feature_contributions``. Uses a small synthetic
feature frame built directly in the ``FEATURES`` schema.
"""

import numpy as np
import pandas as pd
import pytest

import train


def _feature_frame(n=200, seed=0):
    """Synthetic feature matrix in the model's input schema, including the
    awkward cases: NaN ``log_mw`` and an 'Unknown' category."""
    rng = np.random.default_rng(seed)
    regions = ["PJM", "MISO", "West", "Unknown"]
    df = pd.DataFrame({
        "region": rng.choice(regions, n),
        "state": rng.choice(["VA", "TX", "CA", "Unknown"], n),
        "type_clean": rng.choice(["Solar", "Wind", "Battery", "Unknown"], n),
        "service": rng.choice(["NRIS", "ERIS", "Unknown"], n),
        "log_mw": rng.normal(2.0, 0.4, n),
        "q_year_clipped": rng.integers(2000, 2019, n).astype(float),
        "is_hybrid": rng.integers(0, 2, n),
        "ia_executed": rng.integers(0, 2, n),
    })
    df.loc[df.sample(frac=0.15, random_state=seed).index, "log_mw"] = np.nan
    # Label correlated with ia_executed so models have real signal to learn.
    p = np.where(df["ia_executed"] == 1, 0.3, 0.85)
    labels = pd.Series((rng.random(n) < p).astype(int))
    return df[train.FEATURES], labels


@pytest.mark.parametrize("factory", [train.make_rf, train.make_logreg])
def test_fit_predict_with_missing_and_unknown(factory):
    X, y = _feature_frame()
    model = factory()
    model.fit(X, y)  # NaN log_mw + 'Unknown' categories must not raise
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("factory", [train.make_rf, train.make_logreg])
def test_models_handle_unseen_category(factory):
    X, y = _feature_frame()
    model = factory()
    model.fit(X, y)
    novel = X.iloc[[0]].copy()
    novel["region"] = "BRAND_NEW_ISO"  # unseen at fit time
    model.predict_proba(novel)  # handle_unknown='ignore' -> no raise


# --- rf_feature_contributions (SHAP, probability space) -----------------

def test_contributions_are_exactly_additive():
    """baseline + sum(per-feature contributions) must equal P(withdrawn)."""
    X, y = _feature_frame()
    rf = train.make_rf()
    rf.fit(X, y)
    base, contrib = train.rf_feature_contributions(rf, X)
    recon = base + contrib.sum(axis=1).to_numpy()
    proba = rf.predict_proba(X)[:, 1]
    np.testing.assert_allclose(recon, proba, atol=1e-6)


def test_contributions_have_one_column_per_feature():
    X, y = _feature_frame()
    rf = train.make_rf()
    rf.fit(X, y)
    base, contrib = train.rf_feature_contributions(rf, X)
    assert list(contrib.columns) == train.FEATURES
    assert contrib.index.equals(X.index)
    assert isinstance(base, float)
    # Baseline is a probability near the label base rate.
    assert 0.0 <= base <= 1.0


def test_contributions_deterministic():
    X, y = _feature_frame()
    rf = train.make_rf()
    rf.fit(X, y)
    base_a, contrib_a = train.rf_feature_contributions(rf, X)
    base_b, contrib_b = train.rf_feature_contributions(rf, X)
    assert base_a == base_b
    pd.testing.assert_frame_equal(contrib_a, contrib_b)
