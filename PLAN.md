# Project Plan — Queue Score

Withdrawal-risk predictions for active U.S. interconnection-queue projects, served
as a static web app. This file tracks what is built and what comes next.

## Status: built and working

End-to-end pipeline and web app are functional. `model/train.py` trains the
models and regenerates all app data; `app/` is a dependency-free static site.

**Model pipeline (`model/train.py`)**
- Trains a **random forest** (main model, 400 trees) on resolved projects from
  mature queue cohorts (entry years 2000–2018; ~16.4k projects) to avoid censoring
  bias from recent, mostly-unresolved cohorts.
- Leakage-free features: region, state, technology, log capacity, service type,
  hybrid flag, clipped queue year, and a cumulative "reached IA execution"
  milestone flag.
- **No calibration step** — random-forest probabilities are used as-is (read as a
  risk score, not an exact frequency).
- **Per-project explanations** via SHAP (TreeExplainer, exact for the forest):
  baseline + per-feature contributions sum exactly to the predicted probability.
- **Evaluation**: a single temporal holdout (train ≤2014, test 2015–2018); no
  cross-validation (random folds mix queue eras on time-dependent data). A
  **logistic regression** is kept as a comparison baseline.
- Exports `app/data/predictions.json` (per-project calibrated probability + exact
  EBM term contributions), `app/data/model_meta.json` (metrics, importances), and
  the `app/img/*.png` figures.

**Web app (`app/`)**
- `index.html` — searchable/filterable/sortable explorer of all ~8.5k active
  projects; "how many projects ahead of me are real?" tool; per-project
  explanation modal (deep-linkable via `#p<id>`).
- `methodology.html` — censoring rationale + figure, feature/leakage notes,
  calibration, model comparison tables, global importances, limitations.

**Data source (current)**: the LBNL "Queued Up" workbook, data through 2025
(`data/LBNL_Ix_Queue_Data_File_thru2025.xlsx`). Path/sheet are set at the top of
`model/train.py` (`DATA_FILE`, `QUEUE_SHEET`).

## Next steps

### 1. Swap in the latest LBNL data — DONE (2026-06-18)
Moved off the CSV snapshot onto the through-2025 workbook.
- Added an xlsx loader (`load_data` reads the "03. Complete Queue Data" sheet,
  `header=1`) with a `COLUMN_RENAMES` map for edition differences
  (`mw_1`→`mw1`, `IA_phase_clean`→`IA_status_clean`); `excel_date` now also
  accepts real datetimes (the workbook no longer stores Excel serials).
- **Training window extended to 2018** (`TRAIN_YEAR_MAX`, `q_year_clipped`): the
  2018 cohort is now ~83% resolved, the same maturity bar that set the old 2017
  cutoff. Training pool 16,364 projects; 8,513 active projects scored.
- Sanity check surfaced a real change: the 2025 edition's cleaner IA coding means
  **no completed project lacks an executed IA**, so `ia_executed` now perfectly
  separates resolved outcomes and dominates the model — ~70% of active projects
  (those without an IA) read ~99% withdrawal. After reviewing the
  `notebooks/ia_column_2024_vs_2025` comparison, the decision was to **keep
  `ia_executed`** (faithful to the record; strongest real signal) and document the
  overstated-certainty caveat for early-stage projects in the methodology page.
- **Follow-up worth considering** (not blocking): treat the no-IA active majority
  with a censoring-aware approach (e.g., survival / competing-risks model, or a
  separate calibration for the no-IA segment) so early-stage projects aren't all
  pinned at ~99%. This is a modeling change beyond a data swap.

### 2. Use gridstatus for the most current queue data
LBNL is an annual, cleaned snapshot; it lags the live queues. Add
[`gridstatus`](https://github.com/gridstatus/gridstatus) to pull near-real-time
interconnection-queue data directly from the ISOs/RTOs (CAISO, MISO, PJM, SPP,
ERCOT, NYISO, ISO-NE).
- Use gridstatus to refresh the **active project list and statuses** more
  frequently than LBNL publishes, so predictions reflect the current queue.
- Keep **LBNL as the training source** (it has the clean historical outcome
  labels gridstatus lacks); use gridstatus for **scoring** the live active set.
- Build a small harmonization layer mapping gridstatus per-ISO schemas onto the
  feature columns the model expects (region, state, technology, MW, service, IA
  status → `ia_executed`, queue date). Per-ISO field naming is the main effort.
- Goal: a repeatable refresh that re-scores live projects without retraining,
  plus a periodic retrain when a new LBNL edition lands.

### 3. Deployment
Ship the static app and make refreshes reproducible.
- Host `app/` on static hosting (GitHub Pages / Netlify / Cloudflare Pages) —
  no server needed; it only reads the JSON in `app/data/`.
- Add a `requirements.txt`/lockfile and a one-command build (`make data` →
  `python model/train.py`) so the JSON/figures regenerate cleanly.
- Automate the refresh (e.g., a scheduled CI job): pull latest data, retrain or
  re-score, commit regenerated `app/data/` + `app/img/`, redeploy.
- Surface the data vintage in the UI (the `generated` date is already in the
  JSON) so users know how current the predictions are.

## Known limitations to carry forward
Documented in `app/methodology.html`: queue-year extrapolation for post-2017
entrants, early-stage IA conditioning, eventual-vs-dated outcomes, post-training
policy shifts (FERC Order 2023), and absence of project financials/site control.
