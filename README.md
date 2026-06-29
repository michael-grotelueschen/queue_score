# Queue Score

Predicts the probability that each active project in U.S. interconnection queues
will be **withdrawn before reaching commercial operation**, and presents the
predictions in a searchable web app — answering the developer question
*"how many of the projects ahead of me in the queue are real?"*

Built on the LBNL ["Queued Up"](https://emp.lbl.gov/queues) complete
interconnection-request dataset (latest edition, data through 2025).

## Run

```bash
cd app && python3 -m http.server 8742
# open http://localhost:8742
```

The app is fully static — predictions are precomputed into `app/data/`.

## Retrain

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python model/train.py   # auto-downloads the LBNL workbook to data/ if missing
```

The raw LBNL `.xlsx` workbooks are git-ignored; `model/train.py` downloads the one
it needs on first run via `data/fetch_workbooks.py` (pinned URLs, browser
User-Agent + certifi CA bundle). To fetch every edition up front:

```bash
.venv/bin/python data/fetch_workbooks.py
```

When LBNL publishes a new edition, add its direct URL to `WORKBOOKS` in that file.

`model/train.py` regenerates `app/data/predictions.json` (per-project
probabilities + exact SHAP feature contributions), `app/data/model_meta.json`
(evaluation metrics shown on the methodology page), and
`app/img/evaluation.png` (reliability + ROC plots).

## Test

```bash
.venv/bin/pip install -r requirements-dev.txt   # runtime + pytest + notebook tooling
.venv/bin/python -m pytest
```

`tests/` covers data cleaning and missing-data handling (`load_data`,
`build_features`, `excel_date`), the no-leakage guarantee, the SHAP contribution
helper (exact additivity), the metric helpers, and an output contract on the
generated JSON (probabilities in range, explanation faithfulness, model coverage).
Tests that need the LBNL workbook or the generated `app/data/*.json` skip cleanly
when those are absent, so the synthetic-fixture tests run anywhere.

## Model in one paragraph

A **random forest** (400 trees, scikit-learn) trained on 16,364 resolved projects
from queue cohorts 2000–2018 (mature cohorts only, to avoid censoring bias from
recent entrants whose successes are still pending). Features are leakage-free and
observable for active projects: region, state, technology, log capacity, service
type, hybrid flag, queue year, and a "reached IA execution" milestone indicator.
Predicted probabilities are used **as-is** — there is no calibration step — so read
them as a well-ordered risk score more than an exact frequency. Evaluation is a
single **temporal holdout** (train 2000–2014, test 2015–2018); cross-validation is
deliberately avoided because random folds mix queue eras and overstate skill on
time-dependent data. A **logistic regression** is kept as a comparison baseline; on
the holdout the two are close on AUC (RF 0.955, LR 0.957), with the logistic
regression's raw probabilities somewhat better calibrated. Per-project explanations
come from **SHAP** (TreeExplainer, exact for the forest): a baseline plus per-feature
contributions that sum exactly to the predicted probability. Full details and
limitations: `app/methodology.html`.

Note: in the through-2025 data the "reached IA execution" milestone is an
especially dominant predictor — no completed project lacks an executed IA, so the
~70% of active projects without one receive a high withdrawal probability (~95–99%)
with little differentiation. This is faithful to the historical record but
overstates certainty for early-stage projects; see the methodology limitations.

## Layout

```
requirements.txt      runtime deps (pinned); requirements-dev.txt adds tests + notebooks
data/                 LBNL "Queued Up" workbooks (.xlsx, git-ignored) +
                      fetch_workbooks.py (downloads them; pinned URLs)
model/train.py        training + evaluation + explanation + export pipeline
app/index.html        explorer: search/filter, "queue ahead of me" tool,
                      per-project explanation modal (deep-linkable as #p<id>)
app/methodology.html  methodology, random forest vs logistic regression metrics, limitations
app/data/             precomputed predictions + model metadata
app/img/              evaluation figures
notebooks/            eda_training_data — columns/dtypes/nulls, resolution &
                      withdrawal rates by year, an IA-column deep dive, and a
                      time-to-operation boxplot; ia_column_2024_vs_2025 — IA-column
                      comparison across editions (each has a build_*.py generator)
tests/                pytest suite (data cleaning, leakage, calibration, contract)
```
