"""Train the interconnection-queue withdrawal-risk model and export the web app data.

This script is the entire offline pipeline behind the "Queue Score" web
app. It learns, from historical queue outcomes, the probability that an active
interconnection request will be withdrawn before reaching commercial operation,
then writes per-project predictions and model metadata that the static front end
reads directly.

Pipeline (see ``main`` for the orchestration):
  1. Load and filter the LBNL "Queued Up" complete queue dataset (``load_data``).
  2. Derive leakage-free features that are observable for an active project today
     (``build_features``).
  3. Restrict the training population to *resolved* projects (withdrawn or
     operational) from *mature* queue cohorts (entry years 2000-2018). Recent
     cohorts are excluded because their successes have not had time to reach
     operation, so their resolved subset is almost entirely withdrawals and would
     bias the model (this censoring effect is visualized by ``censoring_figure``).
  4. Train a random forest (the main model; ``make_rf``) and a logistic-regression
     baseline for comparison (``make_logreg``). Probabilities are used as-is —
     there is no calibration step.
  5. Evaluate both models on a single temporal holdout — train on <=2014 entrants,
     test on 2015-2018 entrants — the realistic "predict the future" setting.
     There is no cross-validation (it would mix queue eras and overstate skill on
     this time-dependent data).
  6. Refit the production random forest on the full mature window, score every
     active project, and explain each prediction with exact SHAP feature
     contributions (``rf_feature_contributions``) for the per-project breakdown.

Outputs (all under ``app/``):
  - ``data/predictions.json``  per-active-project withdrawal probability,
                               descriptive fields, and SHAP feature contributions.
  - ``data/model_meta.json``   evaluation metrics, global importances, SHAP
                               baseline, and headline figures for the methodology page.
  - ``img/evaluation.png``     reliability diagram + ROC curve (temporal holdout).
  - ``img/censoring.png``      cohort status composition by queue entry year.
"""

import json
import datetime as dt
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
# Latest LBNL "Queued Up" workbook (data through end of 2025, released May 2026).
DATA_FILE = ROOT / "data" / "LBNL_Ix_Queue_Data_File_thru2025.xlsx"
QUEUE_SHEET = "03. Complete Queue Data"
APP_DATA = ROOT / "app" / "data"
APP_IMG = ROOT / "app" / "img"
APP_DATA.mkdir(parents=True, exist_ok=True)
APP_IMG.mkdir(parents=True, exist_ok=True)

# The 2025 edition renamed some columns relative to earlier editions; normalize
# them at load time so the rest of the pipeline uses one stable schema.
COLUMN_RENAMES = {
    "mw_1": "mw1",
    "mw_2": "mw2",
    "mw_3": "mw3",
    "IA_phase_clean": "IA_status_clean",
}

RANDOM_SEED = 42
# Train on cohorts mature enough to be mostly resolved. As of the through-2025
# data, the 2018 cohort is ~83% resolved (the same bar that set the prior 2017
# cutoff), so the window extends one year to 2018.
TRAIN_YEAR_MIN, TRAIN_YEAR_MAX = 2000, 2018
TIME_SPLIT_YEAR = 2014  # train <= this, test = (this, TRAIN_YEAR_MAX]

EXCEL_EPOCH = dt.date(1899, 12, 30)

CAT_FEATURES = ["region", "state", "type_clean", "service"]
NUM_FEATURES = ["log_mw", "q_year_clipped"]
BIN_FEATURES = ["is_hybrid", "ia_executed"]
FEATURES = CAT_FEATURES + NUM_FEATURES + BIN_FEATURES

FEATURE_LABELS = {
    "region": "Region",
    "state": "State",
    "type_clean": "Technology",
    "service": "Service type",
    "log_mw": "Capacity (MW)",
    "q_year_clipped": "Queue entry year",
    "is_hybrid": "Hybrid project",
    "ia_executed": "Interconnection agreement executed",
}

IA_EXECUTED_STATUSES = {"IA Executed", "Construction", "Operational", "Combined"}


def excel_date(value):
    """Normalize a date cell to an ISO date string.

    Handles both representations seen across LBNL editions: the through-2025
    workbook reads dates as real datetimes (pandas ``Timestamp``), while older
    CSV exports carried them as Excel serial numbers (days since 1899-12-30, the
    ``EXCEL_EPOCH`` that accounts for Excel's 1900 leap-year bug).

    Args:
        value: A datetime/``Timestamp``, an Excel serial day count, or NaN/None
            for a missing date.

    Returns:
        ISO-8601 date string ("YYYY-MM-DD"), or None if the input is missing or
        cannot be interpreted as a valid in-range date.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(value).date().isoformat()
    try:
        return (EXCEL_EPOCH + dt.timedelta(days=float(value))).isoformat()
    except (ValueError, OverflowError, TypeError):
        return None


def _ensure_workbook():
    """Make sure ``DATA_FILE`` exists, downloading it if necessary.

    Delegates to ``data/fetch_workbooks.py`` (loaded by path so ``data`` need not
    be a package). This makes the pipeline self-bootstrapping: the large ``.xlsx``
    workbooks are git-ignored, and a fresh checkout fetches them on first run.
    """
    if DATA_FILE.exists():
        return
    fetcher_path = ROOT / "data" / "fetch_workbooks.py"
    spec = importlib.util.spec_from_file_location("fetch_workbooks", fetcher_path)
    fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetcher)
    fetcher.ensure(DATA_FILE)


def load_data():
    """Load the LBNL queue workbook and keep only rows with a usable queue status.

    Reads the "Complete Queue Data" sheet of the through-2025 workbook. That
    sheet has a single navigation-banner row above the real header, so
    ``header=1`` lands on the column names. The string "NA" is treated as
    missing, edition-specific column names are normalized via ``COLUMN_RENAMES``
    (e.g. ``mw_1`` -> ``mw1``, ``IA_phase_clean`` -> ``IA_status_clean``), and
    rows whose ``q_status`` is not one of the four modeled states (notably the
    handful of "unknown" rows) are dropped. The workbook is downloaded
    automatically (via ``data/fetch_workbooks.py``) if it is not already present.

    Returns:
        A DataFrame of the active, withdrawn, operational, and suspended projects
        in the pipeline's normalized column schema. Date columns are datetimes.
    """
    _ensure_workbook()
    queue = pd.read_excel(
        DATA_FILE, sheet_name=QUEUE_SHEET, header=1, na_values=["NA"]
    )
    queue = queue.rename(columns=COLUMN_RENAMES)
    queue = queue[queue["q_status"].isin(["active", "withdrawn", "operational", "suspended"])].copy()
    return queue


def build_features(queue):
    """Derive the model feature matrix from raw queue columns.

    Every feature here is observable for a project that is still active today, so
    the same function is used for both training rows and prediction rows. The two
    leakage-sensitive choices are made deliberately:

      - ``q_year_clipped`` clips the queue entry year to the training window
        [2000, 2018]. Active projects mostly entered after 2017, but the model
        never saw those years in training, so clipping evaluates them at the
        boundary the model understands instead of extrapolating.
      - ``ia_executed`` is a *cumulative milestone* flag: it is true if the
        project has ever reached an executed interconnection agreement (by status
        or by the presence of an IA date). For a resolved project this records a
        milestone it passed before resolving, not its final disposition, so it
        does not leak the withdrawn/operational label.

    Categorical fields are filled with the literal "Unknown" so missingness is its
    own category. ``log_mw`` is log10 of nameplate capacity, with non-positive
    capacities treated as missing (left as NaN for the models to handle).

    Args:
        queue: A DataFrame in the schema returned by ``load_data``.

    Returns:
        A DataFrame indexed identically to ``queue`` with exactly the columns in
        ``FEATURES``.
    """
    features = pd.DataFrame(index=queue.index)
    features["region"] = queue["region"].fillna("Unknown")
    features["state"] = queue["state"].fillna("Unknown")
    features["type_clean"] = queue["type_clean"].fillna("Unknown")
    features["service"] = queue["service"].fillna("Unknown")
    capacity_mw = queue["mw1"].where(queue["mw1"] > 0)
    features["log_mw"] = np.log10(capacity_mw)
    features["q_year_clipped"] = queue["q_year"].clip(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    features["is_hybrid"] = (
        queue["type_clean"].fillna("").str.contains(r"\+") | queue["mw2"].notna()
    ).astype(int)
    features["ia_executed"] = (
        queue["IA_status_clean"].isin(IA_EXECUTED_STATUSES) | queue["ia_date"].notna()
    ).astype(int)
    return features


def make_preprocessor():
    """Build the shared feature preprocessor for both models.

    Categorical and binary features are one-hot encoded (rare categories with
    fewer than 20 occurrences are folded into an "infrequent" bucket and unseen
    categories at predict time are ignored); numeric features are median-imputed
    and standardized. The random forest and the logistic regression reuse this so
    they see exactly the same feature representation as each other.

    Returns:
        A new, unfitted ``ColumnTransformer``.
    """
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                CAT_FEATURES + BIN_FEATURES,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                NUM_FEATURES,
            ),
        ]
    )


def make_rf():
    """Construct the unfitted production model: a random-forest pipeline.

    The random forest is the main model. It is a flexible ensemble that
    discriminates well; its probabilities are used *as-is* (no calibration step).
    ``min_samples_leaf`` is set fairly high to keep leaf probability estimates
    stable. Uses ``make_preprocessor`` for feature handling.

    Returns:
        A new, unfitted scikit-learn ``Pipeline``.
    """
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_logreg():
    """Construct the unfitted logistic-regression baseline pipeline.

    This is the interpretable linear model reported alongside the random forest
    for comparison. It uses ``make_preprocessor`` for feature handling followed by
    an L2-regularized logistic regression.

    Returns:
        A new, unfitted scikit-learn ``Pipeline`` mirroring the forest's feature set.
    """
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def rf_feature_contributions(rf_pipeline, features):
    """Exact additive per-feature contributions for a fitted RF pipeline.

    Uses SHAP's ``TreeExplainer`` (exact for tree ensembles) on the random
    forest, in probability space, then sums the one-hot-encoded columns back onto
    their source feature. The decomposition is exact: for every row,
    ``base_value + sum(contributions) == P(withdrawn)`` predicted by the forest.
    This powers the app's per-project "Why this prediction?" panel.

    Args:
        rf_pipeline: A fitted ``make_rf`` pipeline (``prep`` + ``rf`` steps).
        features: Feature DataFrame in the ``FEATURES`` schema.

    Returns:
        Tuple ``(base_value, contributions)`` where ``base_value`` is the float
        SHAP baseline (≈ the training withdrawal rate) and ``contributions`` is a
        DataFrame with one column per entry in ``FEATURES`` (probability points),
        indexed like ``features``.
    """
    prep = rf_pipeline.named_steps["prep"]
    rf = rf_pipeline.named_steps["rf"]
    encoded = prep.transform(features)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    explainer = shap.TreeExplainer(rf)
    shap_values = np.asarray(explainer.shap_values(encoded))
    base_value = float(np.atleast_1d(explainer.expected_value)[1])
    # Take the positive-class (withdrawn) contributions.
    contrib_encoded = shap_values[:, :, 1] if shap_values.ndim == 3 else np.asarray(shap_values)[1]

    # Map each encoded column ("cat__region_PJM", "num__log_mw", ...) to its
    # source feature, then sum encoded columns onto that feature.
    source = []
    for name in prep.get_feature_names_out():
        body = name.split("__", 1)[1]
        source.append(next(
            (f for f in FEATURES if body == f or body.startswith(f + "_")), None
        ))
    contributions = pd.DataFrame(
        0.0, index=features.index, columns=FEATURES
    )
    for col, feature in enumerate(source):
        if feature is not None:
            contributions[feature] += contrib_encoded[:, col]
    return base_value, contributions


def ece(labels, preds, n_bins=10):
    """Compute Expected Calibration Error using equal-count (quantile) bins.

    Predictions are sorted and split into ``n_bins`` bins of equal size. For each
    bin, the absolute gap between the mean predicted probability and the observed
    outcome rate is measured, and ECE is the sample-size-weighted average of those
    gaps. Lower is better; 0 means perfect calibration on this binning. Equal-count
    bins (rather than equal-width) keep every bin populated even when predictions
    cluster near 0 or 1.

    Args:
        labels: Binary outcomes (array-like of 0/1).
        preds: Predicted probabilities, same length as ``labels``.
        n_bins: Number of quantile bins.

    Returns:
        The expected calibration error as a float in [0, 1].
    """
    order = np.argsort(preds)
    labels, preds = np.asarray(labels)[order], np.asarray(preds)[order]
    bins = np.array_split(np.arange(len(preds)), n_bins)
    error = 0.0
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        error += len(bin_idx) / len(preds) * abs(labels[bin_idx].mean() - preds[bin_idx].mean())
    return error


def metrics(labels, preds):
    """Compute the standard evaluation metric bundle for a set of predictions.

    Args:
        labels: Binary outcomes (1 = withdrawn).
        preds: Predicted withdrawal probabilities aligned to ``labels``.

    Returns:
        A dict with discrimination (``auc``), proper-scoring-rule
        (``brier``, ``log_loss``), and calibration (``ece``) metrics, plus
        ``mean_pred`` and ``base_rate`` whose closeness is a quick calibration
        sanity check. All values are rounded to 4 decimals for JSON export.
    """
    return {
        "auc": round(float(roc_auc_score(labels, preds)), 4),
        "brier": round(float(brier_score_loss(labels, preds)), 4),
        "log_loss": round(float(log_loss(labels, preds)), 4),
        "ece": round(float(ece(labels, preds)), 4),
        "mean_pred": round(float(np.mean(preds)), 4),
        "base_rate": round(float(np.mean(labels)), 4),
    }


def censoring_figure(queue, year_min=2005, year_max=2025):
    """Render the cohort-censoring figure that motivates the training window.

    Produces ``app/img/censoring.png``: two lines vs. queue entry year — the
    **resolution rate** (resolved / cohort) and the **withdrawal rate** (withdrawn
    / resolved). The training window [2000, ``TRAIN_YEAR_MAX``] is shaded.

    The figure makes the survivorship-bias argument visually: for recent cohorts
    the resolution rate falls toward zero (those projects have not had time to
    resolve) while the withdrawal rate among the few that did resolve approaches
    100%. Training on recently resolved projects would therefore learn an
    artificially high withdrawal prior; older cohorts are nearly fully resolved
    and representative.

    Args:
        queue: Full project DataFrame from ``load_data`` (all four statuses needed).
        year_min: First queue entry year to plot.
        year_max: Last queue entry year to plot.

    Side effects:
        Writes the PNG; returns nothing.
    """
    years = list(range(year_min, year_max + 1))
    cohort_rows = queue[queue["q_year"].between(year_min, year_max)]
    status = pd.crosstab(cohort_rows["q_year"], cohort_rows["q_status"])
    status = status.reindex(years, fill_value=0)
    for col in ["operational", "withdrawn", "active", "suspended"]:
        if col not in status:
            status[col] = 0

    total = status[["operational", "withdrawn", "active", "suspended"]].sum(axis=1)
    resolved = status["withdrawn"] + status["operational"]
    resolution_rate = (resolved / total.replace(0, np.nan)) * 100
    withdrawal_rate = (status["withdrawn"] / resolved.replace(0, np.nan)) * 100

    year_positions = np.array(years)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axvspan(year_min - 0.6, TRAIN_YEAR_MAX + 0.5, color="#102a36", alpha=0.06, zorder=0)
    ax.axvline(TRAIN_YEAR_MAX + 0.5, color="#102a36", ls="--", lw=1.2, alpha=0.7)
    ax.text(TRAIN_YEAR_MAX + 0.35, 6, "  training window\n  (mostly resolved)",
            ha="right", va="bottom", fontsize=8.5, color="#102a36")
    ax.plot(year_positions, resolution_rate.to_numpy(), "o-", color="#0b7285",
            lw=1.8, ms=4, label="Resolution rate (resolved / cohort)")
    ax.plot(year_positions, withdrawal_rate.to_numpy(), "s-", color="#c92a2a",
            lw=1.8, ms=4, label="Withdrawal rate (withdrawn / resolved)")
    ax.set_ylim(0, 105)
    ax.set_xlim(year_min - 0.6, year_max + 0.6)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Queue entry year")
    ax.set_title("Resolution and withdrawal rate by queue entry year")
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_positions, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", color="#eef2f5", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="center left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(APP_IMG / "censoring.png", dpi=150)
    plt.close(fig)


def reliability_points(labels, preds, n_bins=10):
    """Compute reliability-diagram points using equal-count bins.

    Predictions are sorted and split into ``n_bins`` equal-size bins; each point
    pairs a bin's mean predicted probability with its observed outcome rate. These
    are the coordinates plotted against the diagonal in the reliability diagram.

    Args:
        labels: Binary outcomes (1 = withdrawn).
        preds: Predicted withdrawal probabilities aligned to ``labels``.
        n_bins: Number of quantile bins.

    Returns:
        A list of ``(mean_predicted, observed_rate, bin_count)`` tuples, ordered
        from lowest to highest predicted probability.
    """
    order = np.argsort(preds)
    labels, preds = np.asarray(labels)[order], np.asarray(preds)[order]
    points = []
    for bin_idx in np.array_split(np.arange(len(preds)), n_bins):
        if len(bin_idx):
            points.append((float(preds[bin_idx].mean()), float(labels[bin_idx].mean()), int(len(bin_idx))))
    return points


def main():
    """Run the full pipeline and write all web-app data and figure artifacts.

    Orchestrates the steps described in the module docstring: load data, build
    features, render the censoring figure, define the training population
    (resolved + mature cohorts), evaluate the random forest (main model) and the
    logistic-regression baseline on a single temporal holdout (no
    cross-validation), refit the production random forest on the full mature
    window, score every active project and explain each prediction with exact
    SHAP feature contributions, and serialize ``predictions.json``,
    ``model_meta.json``, and ``evaluation.png``. Progress and headline numbers are
    printed to stdout. Takes no arguments and returns nothing.
    """
    queue = load_data()
    features_all = build_features(queue)
    censoring_figure(queue)

    resolved = queue["q_status"].isin(["withdrawn", "operational"])
    mature = queue["q_year"].between(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    train_mask = resolved & mature
    labels_all = (queue["q_status"] == "withdrawn").astype(int)

    train_features = features_all[train_mask].reset_index(drop=True)
    train_labels = labels_all[train_mask].reset_index(drop=True)
    train_years = queue.loc[train_mask, "q_year"].reset_index(drop=True)
    print(f"Training pool (resolved, {TRAIN_YEAR_MIN}-{TRAIN_YEAR_MAX}): {len(train_labels)} "
          f"({train_labels.mean():.1%} withdrawn)")

    # ---- Temporal-holdout evaluation (no cross-validation) ----------------
    is_train = train_years <= TIME_SPLIT_YEAR
    is_test = ~is_train
    print(f"Time split: train n={is_train.sum()}, test n={is_test.sum()} "
          f"(test withdrawal rate {train_labels[is_test].mean():.1%})")

    results = {}
    fig_data = {}
    for name, factory in [("rf", make_rf), ("logreg", make_logreg)]:
        model = factory()
        model.fit(
            train_features[is_train].reset_index(drop=True),
            train_labels[is_train].reset_index(drop=True),
        )
        preds = model.predict_proba(train_features[is_test])[:, 1]
        results[name] = {"holdout": metrics(train_labels[is_test], preds)}
        fig_data[name] = {
            "y": train_labels[is_test].to_numpy(),
            "preds": preds,
            "reliability": reliability_points(train_labels[is_test], preds),
        }
        print(name, "holdout:", results[name]["holdout"])

    # ---- Production fit: random forest on the full mature window -----------
    rf_model = make_rf()
    rf_model.fit(train_features, train_labels)

    # ---- Figures: random forest (main) vs logistic regression (baseline) --
    holdout_label = f"{TIME_SPLIT_YEAR + 1}-{TRAIN_YEAR_MAX} holdout"
    model_styles = [("rf", "Random forest", "#6741d9"),
                    ("logreg", "Logistic regression", "#e8590c")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for name, label, color in model_styles:
        points = fig_data[name]["reliability"]
        ax.plot([mean_pred for mean_pred, _, _ in points],
                [observed for _, observed, _ in points],
                "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("Mean predicted withdrawal probability")
    ax.set_ylabel("Observed withdrawal rate")
    ax.set_title(f"Reliability diagram - {holdout_label}")
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, label, color in model_styles:
        fpr, tpr, _ = roc_curve(fig_data[name]["y"], fig_data[name]["preds"])
        auc = results[name]["holdout"]["auc"]
        ax.plot(fpr, tpr, label=f"{label} (AUC {auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC - {holdout_label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(APP_IMG / "evaluation.png", dpi=150)
    plt.close(fig)

    # ---- Predict + explain active projects --------------------------------
    active = queue[queue["q_status"] == "active"].copy()
    active_features = features_all.loc[active.index]
    active_preds = rf_model.predict_proba(active_features)[:, 1]
    base_value, contributions = rf_feature_contributions(rf_model, active_features)

    # Global importances: mean |SHAP contribution| per feature (probability points).
    importances = sorted(
        ((feature, float(contributions[feature].abs().mean())) for feature in FEATURES),
        key=lambda term: -term[1],
    )

    def term_label(feature, row):
        """Render a "<friendly feature name>: <project's value>" label.

        Formats capacity back to MW, queue year as an integer, and binary flags as
        yes/no.

        Args:
            feature: A name from ``FEATURES``.
            row: The project's feature row (a Series from the feature matrix).

        Returns:
            A display string such as "Interconnection agreement executed: no".
        """
        feature_label = FEATURE_LABELS.get(feature, feature)
        value = row.get(feature)
        if feature == "log_mw":
            display_value = "n/a" if pd.isna(value) else f"{10 ** value:,.0f} MW"
        elif feature == "q_year_clipped":
            display_value = "n/a" if pd.isna(value) else f"{int(value)}"
        elif feature in BIN_FEATURES:
            display_value = "yes" if value == 1 else "no"
        else:
            display_value = str(value)
        return f"{feature_label}: {display_value}"

    today = dt.date.today()
    projects = []
    for position, (row_index, row) in enumerate(active.iterrows()):
        feature_row = active_features.loc[row_index]
        terms = sorted(
            ((term_label(feature, feature_row),
              round(float(contributions.loc[row_index, feature]), 4))
             for feature in FEATURES),
            key=lambda term: -abs(term[1]),
        )
        queue_date = excel_date(row["q_date"])
        years_in_queue = None
        if queue_date:
            years_in_queue = round((today - dt.date.fromisoformat(queue_date)).days / 365.25, 1)
        name = row["project_name"]
        if pd.isna(name) or not str(name).strip():
            name = row["poi_name"] if pd.notna(row["poi_name"]) else "(unnamed)"
        projects.append({
            "id": int(row_index),
            "qid": None if pd.isna(row["q_id"]) else str(row["q_id"]),
            "name": str(name),
            "region": str(row["region"]) if pd.notna(row["region"]) else "Unknown",
            "state": str(row["state"]) if pd.notna(row["state"]) else "?",
            "county": str(row["county"]) if pd.notna(row["county"]) else "",
            "type": str(row["type_clean"]) if pd.notna(row["type_clean"]) else "Unknown",
            "mw": None if pd.isna(row["mw1"]) else round(float(row["mw1"]), 1),
            "utility": str(row["utility"]) if pd.notna(row["utility"]) else "",
            "ia_status": str(row["IA_status_clean"]) if pd.notna(row["IA_status_clean"]) else "Unknown",
            "service": str(row["service"]) if pd.notna(row["service"]) else "",
            "q_year": None if pd.isna(row["q_year"]) else int(row["q_year"]),
            "q_date": queue_date,
            "yrs_in_queue": years_in_queue,
            "hybrid": int(feature_row["is_hybrid"]),
            "ia_exec": int(feature_row["ia_executed"]),
            "p": round(float(active_preds[position]), 4),
            "terms": terms,
        })

    with open(APP_DATA / "predictions.json", "w") as out_file:
        json.dump({"generated": today.isoformat(), "projects": projects}, out_file)

    metadata = {
        "generated": today.isoformat(),
        "build": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": "LBNL Queued Up - complete interconnection queue dataset",
        "main_model": "random forest",
        "n_active": len(projects),
        "n_train": int(len(train_labels)),
        "train_window": [TRAIN_YEAR_MIN, TRAIN_YEAR_MAX],
        "time_split_year": TIME_SPLIT_YEAR,
        "train_withdrawal_rate": round(float(train_labels.mean()), 4),
        "baseline": round(base_value, 4),
        "metrics": results,
        "importances": [[term, round(float(importance), 4)] for term, importance in importances],
        "feature_labels": FEATURE_LABELS,
        "active_mean_p": round(float(np.mean(active_preds)), 4),
        "active_expected_survivors": round(float(np.sum(1 - active_preds)), 1),
    }
    with open(APP_DATA / "model_meta.json", "w") as out_file:
        json.dump(metadata, out_file, indent=1)

    print(f"\nExported {len(projects)} active-project predictions.")
    print(f"Mean predicted withdrawal probability: {np.mean(active_preds):.3f}")
    print(f"Expected survivors among active projects: {np.sum(1 - active_preds):.0f}")
    print(f"SHAP baseline: {base_value:.3f}")
    print("Top importances:", importances[:8])


if __name__ == "__main__":
    main()
