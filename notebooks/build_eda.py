"""Assemble and execute the training-data EDA notebook.

Builds ``eda_training_data.ipynb`` (cells defined below), runs it so the tables
and figures are embedded, and writes it next to this script.
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "eda_training_data.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(
    "# Exploratory analysis: training data\n"
    "\n"
    "Profiles the data behind the withdrawal-risk model, loaded exactly as the\n"
    "pipeline sees it (`train.load_data()` — the LBNL *Queued Up* through-2025\n"
    "workbook, with edition column names normalized and only the four modeled\n"
    "statuses kept).\n"
    "\n"
    "Contents:\n"
    "1. Columns, datatypes, and null percentages.\n"
    "2. Resolution rate and withdrawal rate by queue entry year.\n"
    "\n"
    "The model itself trains on the *resolved* subset of *mature* cohorts (queue\n"
    "years 2000–2018); the by-year charts below show why that window is chosen."
))

cells.append(code(
    "import sys\n"
    "sys.path.insert(0, '../model')  # import the pipeline's own loader\n"
    "import train\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "# train.py forces the non-interactive Agg backend on import; re-enable the\n"
    "# inline backend so figures embed in this notebook.\n"
    "%matplotlib inline\n"
    "\n"
    "pd.set_option('display.max_rows', 40)\n"
    "\n"
    "queue = train.load_data()\n"
    "print('Source file :', train.DATA_FILE.name)\n"
    "print('Shape       :', f'{queue.shape[0]:,} rows x {queue.shape[1]} columns')\n"
    "print('Statuses    :', queue['q_status'].value_counts().to_dict())\n"
    "print('Train window:', f'resolved + q_year {train.TRAIN_YEAR_MIN}-{train.TRAIN_YEAR_MAX}')\n"
))

cells.append(md(
    "## 1. Columns, datatypes, and null percentages\n"
    "One row per column, sorted by null percentage (most-missing first). "
    "`n_unique` is included as a quick cardinality check."
))

cells.append(code(
    "schema = pd.DataFrame({\n"
    "    'dtype': queue.dtypes.astype(str),\n"
    "    'non_null': queue.notna().sum(),\n"
    "    'null_pct': (queue.isna().mean() * 100).round(1),\n"
    "    'n_unique': queue.nunique(dropna=True),\n"
    "})\n"
    "schema.index.name = 'column'\n"
    "schema = schema.sort_values('null_pct', ascending=False)\n"
    "print(f'{len(schema)} columns total\\n')\n"
    "display(schema)\n"
))

cells.append(md(
    "## 2. Resolution rate and withdrawal rate by year\n"
    "\n"
    "For each queue entry year:\n"
    "\n"
    "- **Resolution rate** = resolved / total, where *resolved* = withdrawn +\n"
    "  operational. Low values mean the cohort is still largely pending (censored).\n"
    "- **Withdrawal rate (among resolved)** = withdrawn / resolved — of the projects\n"
    "  that have reached an outcome, the share that were withdrawn.\n"
    "- **Withdrawal share of cohort** = withdrawn / total, shown for reference."
))

cells.append(code(
    "byyear = queue.dropna(subset=['q_year']).copy()\n"
    "byyear['q_year'] = byyear['q_year'].astype(int)\n"
    "byyear = byyear[byyear['q_year'].between(2000, byyear['q_year'].max())]\n"
    "\n"
    "ct = pd.crosstab(byyear['q_year'], byyear['q_status'])\n"
    "for col in ['withdrawn', 'operational', 'active', 'suspended']:\n"
    "    if col not in ct:\n"
    "        ct[col] = 0\n"
    "\n"
    "rates = pd.DataFrame(index=ct.index)\n"
    "rates['n'] = ct.sum(axis=1)\n"
    "rates['resolved'] = ct['withdrawn'] + ct['operational']\n"
    "rates['withdrawn'] = ct['withdrawn']\n"
    "rates['operational'] = ct['operational']\n"
    "rates['resolution_rate'] = rates['resolved'] / rates['n']\n"
    "rates['withdrawal_rate_among_resolved'] = (\n"
    "    rates['withdrawn'] / rates['resolved'].replace(0, np.nan)\n"
    ")\n"
    "rates['withdrawal_share_of_cohort'] = rates['withdrawn'] / rates['n']\n"
    "display(rates.round(3))\n"
))

cells.append(code(
    "years = rates.index.to_numpy()\n"
    "lo, hi = train.TRAIN_YEAR_MIN, train.TRAIN_YEAR_MAX\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)\n"
    "\n"
    "# Panel A: resolution rate (line) over cohort size (bars).\n"
    "ax = axes[0]\n"
    "ax.bar(years, rates['n'], color='#dce3e8', label='cohort size (n)')\n"
    "ax.set_ylabel('Cohort size'); ax.set_xlabel('Queue entry year')\n"
    "ax.axvspan(lo - 0.5, hi + 0.5, color='#102a36', alpha=0.05, zorder=0)\n"
    "axr = ax.twinx()\n"
    "axr.plot(years, rates['resolution_rate'] * 100, 'o-', color='#0b7285',\n"
    "         label='resolution rate')\n"
    "axr.set_ylabel('Resolution rate (%)', color='#0b7285'); axr.set_ylim(0, 105)\n"
    "axr.tick_params(axis='y', labelcolor='#0b7285')\n"
    "ax.set_title('Resolution rate by queue entry year')\n"
    "\n"
    "# Panel B: withdrawal rate among resolved (line) over resolved count (bars).\n"
    "ax = axes[1]\n"
    "ax.bar(years, rates['resolved'], color='#dce3e8', label='resolved count')\n"
    "ax.set_ylabel('Resolved count'); ax.set_xlabel('Queue entry year')\n"
    "ax.axvspan(lo - 0.5, hi + 0.5, color='#102a36', alpha=0.05, zorder=0)\n"
    "axr = ax.twinx()\n"
    "axr.plot(years, rates['withdrawal_rate_among_resolved'] * 100, 'o-',\n"
    "         color='#c92a2a', label='withdrawal rate (among resolved)')\n"
    "axr.set_ylabel('Withdrawal rate among resolved (%)', color='#c92a2a')\n"
    "axr.set_ylim(0, 105); axr.tick_params(axis='y', labelcolor='#c92a2a')\n"
    "ax.set_title('Withdrawal rate by queue entry year')\n"
    "\n"
    "fig.suptitle('Shaded band = model training window '\n"
    "             f'({lo}\\u2013{hi})', fontsize=9, y=1.02)\n"
    "fig.tight_layout(); plt.show()\n"
))

cells.append(md(
    "### Takeaway\n"
    "The **resolution rate** sits near 100% for older cohorts and falls off sharply\n"
    "for recent years — those projects simply have not had time to resolve, which is\n"
    "the censoring the training window guards against. Among the projects that *have*\n"
    "resolved, the **withdrawal rate** is high throughout and climbs toward ~99% for\n"
    "the most recent cohorts (their few resolved projects are almost all early\n"
    "withdrawals). Training on the resolved subset of mature cohorts (2000–2018,\n"
    "shaded) keeps the labels representative rather than dominated by this artifact."
))

cells.append(md(
    "## 3. Interconnection agreement columns in detail\n"
    "\n"
    "The IA milestone drives the model's strongest feature, `ia_executed`. Three raw\n"
    "columns describe it: `IA_status_clean` (normalized phase), `ia_date` (execution\n"
    "date), and `IA_phase_raw` (the high-cardinality source text). Below are the\n"
    "value distributions plus resolution and withdrawal rates for each group,\n"
    "**keeping nulls as their own category** — important because `IA_status_clean` is\n"
    "~19% null and `ia_date` is ~87% null, and those null groups are not benign."
))

cells.append(code(
    "ia_cols = ['IA_status_clean', 'ia_date', 'IA_phase_raw']\n"
    "print('IA column null %:')\n"
    "print((queue[ia_cols].isna().mean() * 100).round(1).to_string())\n"
    "\n"
    "print('\\nIA_status_clean — value counts (nulls kept):')\n"
    "print(queue['IA_status_clean'].value_counts(dropna=False).to_string())\n"
    "\n"
    "print(f\"\\nIA_phase_raw — top 10 of {queue['IA_phase_raw'].nunique():,} raw values:\")\n"
    "print(queue['IA_phase_raw'].value_counts(dropna=False).head(10).to_string())\n"
    "\n"
    "# Resolution / withdrawal rates per group, KEEPING nulls as their own category\n"
    "# (groupby dropna=False). 'resolved' = withdrawn or operational.\n"
    "ia = queue.copy()\n"
    "ia['resolved'] = ia['q_status'].isin(['withdrawn', 'operational'])\n"
    "ia['withdrawn'] = (ia['q_status'] == 'withdrawn').astype(int)\n"
    "ia['ia_executed'] = (\n"
    "    ia['IA_status_clean'].isin(train.IA_EXECUTED_STATUSES) | ia['ia_date'].notna()\n"
    ").astype(int)\n"
    "ia['ia_date_present'] = ia['ia_date'].notna()\n"
    "\n"
    "def ia_rates(column):\n"
    "    g = ia.groupby(column, dropna=False)\n"
    "    out = pd.DataFrame({'n': g.size(),\n"
    "                        'resolved': g['resolved'].sum(),\n"
    "                        'withdrawn': g['withdrawn'].sum()})\n"
    "    out['resolution_rate'] = out['resolved'] / out['n']\n"
    "    out['withdrawal_rate_among_resolved'] = (\n"
    "        out['withdrawn'] / out['resolved'].replace(0, np.nan)\n"
    "    )\n"
    "    out['withdrawal_share_of_total'] = out['withdrawn'] / out['n']\n"
    "    return out.sort_values('n', ascending=False)\n"
    "\n"
    "print('\\nResolution / withdrawal rates by IA_status_clean (nulls kept):')\n"
    "display(ia_rates('IA_status_clean').round(3))\n"
    "print('By ia_executed flag (the model feature):')\n"
    "display(ia_rates('ia_executed').round(3))\n"
    "print('By ia_date present vs absent:')\n"
    "display(ia_rates('ia_date_present').round(3))\n"
))

cells.append(code(
    "# Visualize both rates per IA_status_clean group, nulls included, ordered by\n"
    "# withdrawal rate so 'IA Executed' (the only group that ever completes) stands out.\n"
    "t = ia_rates('IA_status_clean').sort_values('withdrawal_rate_among_resolved')\n"
    "labels = ['(null)' if pd.isna(i) else str(i) for i in t.index]\n"
    "y = np.arange(len(t))\n"
    "h = 0.4\n"
    "fig, ax = plt.subplots(figsize=(10, 6))\n"
    "ax.barh(y + h / 2, t['resolution_rate'] * 100, height=h,\n"
    "        color='#0b7285', label='resolution rate')\n"
    "ax.barh(y - h / 2, t['withdrawal_rate_among_resolved'] * 100, height=h,\n"
    "        color='#c92a2a', label='withdrawal rate (among resolved)')\n"
    "ax.set_yticks(y); ax.set_yticklabels(labels)\n"
    "ax.set_xlim(0, 100); ax.set_xlabel('Rate (%)')\n"
    "ax.set_title('Resolution & withdrawal rates by IA_status_clean (nulls kept)')\n"
    "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
))

cells.append(md(
    "### What the IA breakdown shows\n"
    "- **Only `IA Executed` ever completes.** Its withdrawal-rate-among-resolved is\n"
    "  ~35%; every other phase — and the **null** group — is **100% withdrawal among\n"
    "  resolved**. No project reaches operation without an executed IA.\n"
    "- **Nulls are not benign.** The ~7,200 null `IA_status_clean` rows resolve at a\n"
    "  high rate and, when resolved, are *entirely* withdrawals — they behave like the\n"
    "  early, pre-IA phases. Dropping nulls would discard a large, informative slice,\n"
    "  so the rates above keep them.\n"
    "- Consequently the derived **`ia_executed`** flag separates resolved outcomes\n"
    "  almost perfectly (no IA → 100% withdrawn), which is why it dominates the model\n"
    "  and pins no-IA active projects near ~99% (see the methodology limitations)."
))

cells.append(md(
    "## 4. Time to reach operation by queue entry year\n"
    "\n"
    "For projects that reached commercial operation, the time from queue entry\n"
    "(`q_date`) to COD (`on_date`), in years, boxplotted by queue entry year. Only\n"
    "operational projects have an `on_date`, and a handful of records with `on_date`\n"
    "*before* `q_date` (data-entry errors) are dropped.\n"
    "\n"
    "**Mind the right-censoring:** a cohort's box only contains projects that have\n"
    "*already* finished, so recent years are biased toward fast builders (slow ones\n"
    "haven't reached COD yet). The plot is therefore limited to the 2000–2020 window,\n"
    "where cohorts are largely complete; even there the latest couple of years lean\n"
    "mildly optimistic."
))

cells.append(code(
    "op = queue[(queue['q_status'] == 'operational')\n"
    "           & queue['on_date'].notna() & queue['q_date'].notna()].copy()\n"
    "op['years_to_cod'] = (op['on_date'] - op['q_date']).dt.days / 365.25\n"
    "n_bad = (op['years_to_cod'] < 0).sum()\n"
    "op = op[op['years_to_cod'] >= 0]            # drop on_date-before-q_date errors\n"
    "op['q_year'] = op['q_year'].astype(int)\n"
    "print(f'operational projects with a build-out time: {len(op):,} '\n"
    "      f'(dropped {n_bad} with on_date before q_date)')\n"
    "\n"
    "years = [y for y in range(2000, 2021) if (op['q_year'] == y).sum() >= 20]\n"
    "data = [op.loc[op['q_year'] == y, 'years_to_cod'].to_numpy() for y in years]\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 5))\n"
    "bp = ax.boxplot(data, positions=years, widths=0.65, patch_artist=True,\n"
    "                medianprops=dict(color='#102a36', lw=1.4),\n"
    "                flierprops=dict(marker='.', markersize=3, alpha=0.35, mec='#888'))\n"
    "for box in bp['boxes']:\n"
    "    box.set(facecolor='#a5d8e6', edgecolor='#0b7285')\n"
    "overall = op.loc[op['q_year'].between(2000, 2020), 'years_to_cod'].median()\n"
    "ax.axhline(overall, color='#c92a2a', ls='--', lw=1,\n"
    "           label=f'overall median {overall:.1f} yr')\n"
    "ax.set_xlabel('Queue entry year')\n"
    "ax.set_ylabel('Years from queue entry to commercial operation')\n"
    "ax.set_title('Time to reach operation by queue entry year (operational projects)')\n"
    "ax.set_xticks(years); ax.set_xticklabels(years, rotation=45, ha='right', fontsize=8)\n"
    "ax.grid(axis='y', color='#eef2f5', lw=0.8); ax.set_axisbelow(True)\n"
    "ax.legend(fontsize=8.5, loc='upper right')\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "print('\\nYears-to-COD summary (operational, q_year 2000\\u20132020):')\n"
    "print(op.loc[op['q_year'].between(2000, 2020), 'years_to_cod']\n"
    "        .describe(percentiles=[.1, .5, .9]).round(2).to_string())\n"
))

cells.append(md(
    "### Takeaway\n"
    "Median build-out is **~3.5 years** and fairly stable across cohorts, with a long\n"
    "upper tail (90th percentile ~8 years) — consistent with the ~3.5-yr median the\n"
    "methodology cites for why recent cohorts are still censored. The dip in the latest\n"
    "in-window years reflects that censoring (slow builders haven't finished), not\n"
    "genuinely faster projects."
))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "fable_venv", "language": "python", "name": "fable_venv",
}

ep = ExecutePreprocessor(timeout=300, kernel_name="fable_venv")
ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, OUT)
print("wrote and executed:", OUT)
