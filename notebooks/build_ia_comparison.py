"""Assemble and execute the IA-column comparison notebook.

Builds ``ia_column_2024_vs_2025.ipynb`` (cells defined below), runs it so the
tables and figures are embedded, and writes it next to this script.
"""

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
OUT = HERE / "ia_column_2024_vs_2025.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(
    "# IA column: 2024 edition vs. 2025 edition\n"
    "\n"
    "**Why this notebook exists.** When we swapped the model onto the latest LBNL "
    "*Queued Up* workbook (data through 2025), the `ia_executed` feature began "
    "*perfectly* separating outcomes: in the training pool every completed "
    "(operational) project had an executed IA, so a project **without** an IA is, "
    "historically, a guaranteed withdrawal. That collapsed the model onto a single "
    "rule and pinned ~70% of active projects at 0.999.\n"
    "\n"
    "This notebook compares the **interconnection-agreement status column** between "
    "the two editions to show *what changed in the data* and decide how to treat the "
    "feature.\n"
    "\n"
    "- **2024 edition** (`...thru2024_v2.xlsx`): IA column is `IA_status_clean`.\n"
    "- **2025 edition** (`...thru2025.xlsx`): IA column renamed to `IA_phase_clean`.\n"
))

cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "pd.set_option('display.max_columns', 30)\n"
    "\n"
    "# Both editions live in the project's data/ directory (paths are relative to\n"
    "# this notebook under notebooks/).\n"
    "DATA = '../data'\n"
    "F2024 = f'{DATA}/LBNL_Ix_Queue_Data_File_thru2024_v2.xlsx'\n"
    "F2025 = f'{DATA}/LBNL_Ix_Queue_Data_File_thru2025.xlsx'\n"
    "SHEET = '03. Complete Queue Data'\n"
    "\n"
    "# Same definition the model uses (model/train.py).\n"
    "IA_EXECUTED_STATUSES = {'IA Executed', 'Construction', 'Operational', 'Combined'}\n"
    "MODELED = ['active', 'withdrawn', 'operational', 'suspended']\n"
    "\n"
    "def load(path, ia_col):\n"
    "    df = pd.read_excel(path, sheet_name=SHEET, header=1, na_values=['NA'])\n"
    "    df = df[df['q_status'].isin(MODELED)].copy()\n"
    "    # Normalize the IA column name so both editions are comparable.\n"
    "    df['ia_clean'] = df[ia_col]\n"
    "    # Reproduce the model's cumulative 'reached IA execution' flag.\n"
    "    df['ia_executed'] = (\n"
    "        df['ia_clean'].isin(IA_EXECUTED_STATUSES) | df['ia_date'].notna()\n"
    "    ).astype(int)\n"
    "    df['resolved'] = df['q_status'].isin(['withdrawn', 'operational'])\n"
    "    df['withdrawn'] = (df['q_status'] == 'withdrawn').astype(int)\n"
    "    return df\n"
    "\n"
    "df24 = load(F2024, 'IA_status_clean')\n"
    "df25 = load(F2025, 'IA_phase_clean')\n"
    "print('2024 rows:', len(df24), '| 2025 rows:', len(df25))\n"
))

cells.append(md(
    "## 1. IA category distribution\n"
    "How the IA-status categories themselves shifted. Note the 2024 column carries "
    "an explicit `Operational` value; the 2025 `IA_phase_clean` does not."
))

cells.append(code(
    "order = sorted(set(df24['ia_clean'].dropna().unique()) |\n"
    "               set(df25['ia_clean'].dropna().unique()))\n"
    "def share(df):\n"
    "    return (df['ia_clean'].value_counts(normalize=True)\n"
    "            .reindex(order).fillna(0) * 100)\n"
    "tbl = pd.DataFrame({'2024 %': share(df24), '2025 %': share(df25)}).round(1)\n"
    "display(tbl)\n"
    "\n"
    "ax = tbl.plot.barh(figsize=(9, 6), color=['#adb5bd', '#0b7285'])\n"
    "ax.set_xlabel('Share of projects (%)'); ax.set_ylabel('')\n"
    "ax.set_title('IA-status category mix, 2024 vs 2025'); ax.invert_yaxis()\n"
    "plt.tight_layout(); plt.show()\n"
))

cells.append(md(
    "## 2. IA category vs. final outcome\n"
    "For each edition, the outcome mix within each IA category (row-normalized). "
    "This shows which IA categories lead to completion vs. withdrawal."
))

cells.append(code(
    "def outcome_heat(df, title, ax):\n"
    "    ct = pd.crosstab(df['ia_clean'], df['q_status'])\n"
    "    for c in ['operational', 'withdrawn', 'active', 'suspended']:\n"
    "        if c not in ct: ct[c] = 0\n"
    "    ct = ct[['operational', 'withdrawn', 'active', 'suspended']]\n"
    "    frac = ct.div(ct.sum(axis=1), axis=0).reindex(order).fillna(0)\n"
    "    im = ax.imshow(frac.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=1)\n"
    "    ax.set_xticks(range(4)); ax.set_xticklabels(frac.columns, rotation=30, ha='right')\n"
    "    ax.set_yticks(range(len(frac))); ax.set_yticklabels(frac.index, fontsize=8)\n"
    "    ax.set_title(title)\n"
    "    for i in range(frac.shape[0]):\n"
    "        for j in range(frac.shape[1]):\n"
    "            v = frac.values[i, j]\n"
    "            if v > 0.01:\n"
    "                ax.text(j, i, f'{v:.0%}', ha='center', va='center', fontsize=7)\n"
    "    return im\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)\n"
    "outcome_heat(df24, '2024: outcome mix by IA_status_clean', axes[0])\n"
    "im = outcome_heat(df25, '2025: outcome mix by IA_phase_clean', axes[1])\n"
    "fig.colorbar(im, ax=axes, shrink=0.7, label='share of category')\n"
    "plt.show()\n"
))

cells.append(md(
    "## 3. The crux: do completed projects ever lack an IA?\n"
    "The model trains on **resolved** projects. The feature breaks if *completed* "
    "projects always have an IA, because then 'no IA' becomes a perfect predictor "
    "of withdrawal. Here is the IA-status mix of **operational** projects in each "
    "edition, and the share of completed projects flagged `ia_executed == 0`."
))

cells.append(code(
    "op24 = df24[df24['q_status'] == 'operational']\n"
    "op25 = df25[df25['q_status'] == 'operational']\n"
    "comp = pd.DataFrame({\n"
    "    '2024 %': op24['ia_clean'].value_counts(normalize=True).reindex(order).fillna(0) * 100,\n"
    "    '2025 %': op25['ia_clean'].value_counts(normalize=True).reindex(order).fillna(0) * 100,\n"
    "}).round(1)\n"
    "display(comp[(comp.T != 0).any()])\n"
    "\n"
    "print('Share of OPERATIONAL (completed) projects with ia_executed == 0:')\n"
    "print(f'  2024: {(op24[\"ia_executed\"] == 0).mean():.1%}')\n"
    "print(f'  2025: {(op25[\"ia_executed\"] == 0).mean():.1%}   <- drives the perfect separation')\n"
))

cells.append(md(
    "## 4. Resulting separation of the `ia_executed` feature\n"
    "Withdrawal rate among **resolved** projects, split by the `ia_executed` flag. "
    "A bar reaching 100% for `no IA` means perfect separation — exactly the "
    "degeneracy that pins active no-IA projects at ~0.999."
))

cells.append(code(
    "def sep(df):\n"
    "    r = df[df['resolved']]\n"
    "    return pd.Series({\n"
    "        'no IA (ia_executed=0)': r.loc[r['ia_executed'] == 0, 'withdrawn'].mean(),\n"
    "        'has IA (ia_executed=1)': r.loc[r['ia_executed'] == 1, 'withdrawn'].mean(),\n"
    "    }) * 100\n"
    "sep_tbl = pd.DataFrame({'2024': sep(df24), '2025': sep(df25)}).round(1)\n"
    "display(sep_tbl)\n"
    "\n"
    "ax = sep_tbl.T.plot.bar(figsize=(7, 4.5), color=['#c92a2a', '#2b8a3e'], rot=0)\n"
    "ax.set_ylabel('Withdrawal rate among resolved (%)'); ax.set_ylim(0, 105)\n"
    "ax.axhline(100, ls='--', lw=1, color='#888')\n"
    "ax.set_title('Does \"no IA\" perfectly predict withdrawal?')\n"
    "ax.legend(title='IA flag'); plt.tight_layout(); plt.show()\n"
    "\n"
    "n24 = (df24['resolved'] & (df24['ia_executed'] == 0)).sum()\n"
    "comp24 = ((df24['q_status'] == 'operational') & (df24['ia_executed'] == 0)).sum()\n"
    "print(f'2024: of {n24} resolved no-IA projects, {comp24} actually completed '\n"
    "      f'({comp24/n24:.1%}) -> no perfect separation')\n"
    "n25 = (df25['resolved'] & (df25['ia_executed'] == 0)).sum()\n"
    "comp25 = ((df25['q_status'] == 'operational') & (df25['ia_executed'] == 0)).sum()\n"
    "print(f'2025: of {n25} resolved no-IA projects, {comp25} actually completed '\n"
    "      f'({comp25/n25:.1%}) -> perfect separation')\n"
))

cells.append(md(
    "## Takeaway\n"
    "If 2024's *completed-but-no-IA* projects (section 3) were a **data-coding "
    "artifact** that the 2025 edition cleaned up, then 'no IA ⇒ withdrawal' is a "
    "genuine structural fact and keeping the feature is defensible. If instead "
    "those 2024 cases were **real completions that genuinely lacked a recorded IA**, "
    "the 2025 perfect separation is mostly tighter bookkeeping and the feature now "
    "overstates certainty for early-stage active projects. Sections 2–3 are where to "
    "look: inspect *which* 2024 categories the completed-no-IA projects fall into."
))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "fable_venv", "language": "python", "name": "fable_venv",
}

ep = ExecutePreprocessor(timeout=300, kernel_name="fable_venv")
ep.preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, OUT)
print("wrote and executed:", OUT)
