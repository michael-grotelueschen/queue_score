"""Smoke test for ``censoring_figure`` -- it should write a non-empty PNG and not
leak open matplotlib figures, including when some cohort years are empty."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train


def _synthetic_queue(n=400, seed=0):
    rng = np.random.default_rng(seed)
    # Deliberately leave a gap year (2012) empty to exercise the reindex /
    # replace(0, NaN) branches.
    years = rng.choice([y for y in range(2008, 2020) if y != 2012], n)
    return pd.DataFrame({
        "q_year": years,
        "q_status": rng.choice(
            ["withdrawn", "operational", "active", "suspended"], n,
            p=[0.6, 0.15, 0.2, 0.05]),
    })


def test_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "APP_IMG", tmp_path)
    before = len(plt.get_fignums())
    train.censoring_figure(_synthetic_queue(), year_min=2008, year_max=2019)
    out = tmp_path / "censoring.png"
    assert out.exists() and out.stat().st_size > 0
    # No figure left open (the function calls plt.close).
    assert len(plt.get_fignums()) == before
