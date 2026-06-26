"""Shared pytest fixtures and import setup for the model test suite.

``model/train.py`` is a script, not an installed package, so we add the ``model``
directory to ``sys.path`` and import it as ``train``. Importing it runs only
module-level setup (constants + ``mkdir`` of the app output dirs); ``main()`` runs
solely under ``__main__`` and is never triggered by import.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

import train  # noqa: E402  (import after sys.path tweak)

# The real LBNL workbook used by the integration tests.
REAL_DATA = Path(train.DATA_FILE)
PRED_JSON = ROOT / "app" / "data" / "predictions.json"
META_JSON = ROOT / "app" / "data" / "model_meta.json"


@pytest.fixture
def train_module():
    """The imported ``train`` module under test."""
    return train


def _raw_row(**overrides):
    """One raw-queue record with every column ``build_features`` reads.

    Defaults describe a plain, fully-populated solar project; pass overrides to
    exercise a specific cleaning/missing-data path. Columns mirror the schema
    returned by ``load_data`` (the subset the feature builder consumes, plus the
    outcome columns used by the leakage guard).
    """
    row = {
        "q_status": "withdrawn",
        "region": "PJM",
        "state": "VA",
        "type_clean": "Solar",
        "service": "NRIS",
        "mw1": 100.0,
        "mw2": np.nan,
        "q_year": 2015,
        "IA_status_clean": "Feasibility Study",
        "ia_date": np.nan,
        # Outcome columns: present so the leakage guard can mutate them.
        "wd_date": 43000.0,
        "on_date": np.nan,
    }
    row.update(overrides)
    return row


@pytest.fixture
def make_raw():
    """Factory returning a single-column-complete raw queue ``DataFrame``.

    Usage: ``make_raw([{...overrides...}, {...}])`` -> DataFrame with one row per
    dict, each filled out by ``_raw_row``.
    """
    def _make(rows):
        return pd.DataFrame([_raw_row(**r) for r in rows])

    return _make
