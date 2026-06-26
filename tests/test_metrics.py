"""Tests for ``ece``, ``metrics``, and ``reliability_points``."""

import numpy as np
import pytest

import train

EXPECTED_METRIC_KEYS = {"auc", "brier", "log_loss", "ece", "mean_pred", "base_rate"}


# --- ece -----------------------------------------------------------------

def test_ece_near_zero_for_calibrated():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # outcomes match probabilities
    assert train.ece(y, p, n_bins=10) < 0.03


def test_ece_large_for_miscalibrated():
    # Predict 0.5 everywhere while the true rate is ~0.9 -> big gap.
    y = np.ones(1000, dtype=int)
    y[:100] = 0
    p = np.full(1000, 0.5)
    assert train.ece(y, p, n_bins=10) > 0.3


def test_ece_handles_tiny_sample_without_zero_division():
    y = np.array([0, 1, 1])
    p = np.array([0.2, 0.6, 0.6])
    val = train.ece(y, p, n_bins=10)  # more bins than rows
    assert np.isfinite(val) and val >= 0


# --- metrics -------------------------------------------------------------

def test_metrics_keys_and_types():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    m = train.metrics(y, p)
    assert set(m) == EXPECTED_METRIC_KEYS
    assert all(isinstance(v, float) for v in m.values())
    # Rounded to 4 decimals on export.
    assert all(round(v, 4) == v for v in m.values())


def test_metrics_perfect_separation_auc_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert train.metrics(y, p)["auc"] == 1.0


def test_metrics_mean_pred_matches_base_rate_when_calibrated():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    m = train.metrics(y, p)
    assert abs(m["mean_pred"] - m["base_rate"]) < 0.02


# --- reliability_points --------------------------------------------------

def test_reliability_points_structure_and_counts():
    rng = np.random.default_rng(3)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    pts = train.reliability_points(y, p, n_bins=10)
    assert 1 <= len(pts) <= 10
    # Tuples of (mean_pred, observed, count); counts cover all rows.
    assert all(len(t) == 3 for t in pts)
    assert sum(t[2] for t in pts) == 1000
    # Ordered by predicted probability.
    means = [t[0] for t in pts]
    assert means == sorted(means)
    assert all(0 <= t[0] <= 1 and 0 <= t[1] <= 1 for t in pts)
