"""Output-contract tests against the generated app data.

These validate the JSON the static front end consumes. They are skipped if the
pipeline has not been run yet (the files are produced by ``python model/train.py``).
"""

import json

import pytest

from conftest import META_JSON, PRED_JSON

pytestmark = pytest.mark.skipif(
    not (PRED_JSON.exists() and META_JSON.exists()),
    reason="run `python model/train.py` to generate app/data/*.json first",
)


@pytest.fixture(scope="module")
def predictions():
    with open(PRED_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def meta():
    with open(META_JSON) as f:
        return json.load(f)


def test_valid_json_no_nan_or_infinity():
    # Strict JSON forbids NaN/Infinity; the front end's JSON.parse would choke.
    for path in (PRED_JSON, META_JSON):
        with open(path) as f:
            json.loads(f.read(), parse_constant=_reject)


def _reject(token):
    raise AssertionError(f"non-finite constant {token!r} in JSON output")


def test_probabilities_in_unit_interval(predictions):
    for proj in predictions["projects"]:
        assert 0.0 <= proj["p"] <= 1.0


def test_explanation_is_faithful(meta, predictions):
    """baseline + sum(SHAP contributions) must reconstruct the probability p."""
    baseline = meta["baseline"]
    for proj in predictions["projects"][:1000]:
        recon = baseline + sum(value for _, value in proj["terms"])
        assert recon == pytest.approx(proj["p"], abs=0.005)


def test_active_count_consistent(meta, predictions):
    assert meta["n_active"] == len(predictions["projects"])


def test_yrs_in_queue_matches_date_presence(predictions):
    for proj in predictions["projects"]:
        if proj["q_date"] is None:
            assert proj["yrs_in_queue"] is None
        else:
            assert proj["yrs_in_queue"] is not None
            assert proj["yrs_in_queue"] >= 0


def test_metrics_block_has_both_models_holdout(meta):
    for model in ("rf", "logreg"):
        assert "holdout" in meta["metrics"][model]
    # No cross-validation / calibrated variants remain.
    assert set(meta["metrics"].keys()) == {"rf", "logreg"}
    assert "intercept" not in meta and "baseline" in meta
