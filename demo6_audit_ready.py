# %% [markdown]
# # Demo 6 — Audit-Ready ML for the EU AI Act
#
# ### Chapter 6 of the open tabular stack: *the artifact factory*
#
# > **Compliance is a stack property. If your stack doesn't produce the artifacts, your stack is broken.**
#
# August 2026 puts every credit, employment, and insurance model under **Annex IV** of the EU AI Act. The auditor wants performance, calibration, group fairness, and a monitoring plan — written down, in one place, signed off.
#
# This notebook runs a regulated use case (income/credit-screening proxy) through `skrub` + `TabICL`, then assembles the **Annex IV technical documentation draft** as a single PDF the CRO can actually read.
#
# | Annex IV section | Skore primitive |
# |---|---|
# | §2 — performance metrics | `EstimatorReport.metrics.summarize()` |
# | §2 — calibration | `metrics.roc()` / `metrics.precision_recall()` |
# | §3 — group performance | subgroup table (this notebook) |
# | §3 — monitoring | drift hooks (placeholder) |
#
# ---
#
# **Stack used:**
# - `skrub` — `TableVectorizer`
# - `tabicl` — `TabICLClassifier`
# - `skore` — `EstimatorReport`, Hub project
# - `playwright` — HTML → PDF for the auditor packet

# %%
import os
from pathlib import Path

# Load SKORE_HUB_API_KEY from .env at the repo root.
_repo_root = Path.cwd()
while not (_repo_root / ".env").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
_env_path = _repo_root / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

import skore
if "SKORE_HUB_API_KEY" in os.environ:
    skore.login()


# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 110

# :probabl. brand palette
BLUE = "#1E22AA"
ORANGE = "#F68D2E"
GRAY = "#94A3B8"
DARK = "#1A1A2E"

# %% [markdown]
# ## 1. Adult census — a credit / employment-screening proxy
#
# The classic UCI Adult dataset. Sensitive attributes (`sex`, `race`, `age`) are present — exactly what makes it a useful proxy for high-risk AI Act use cases (credit scoring, hiring shortlists, insurance underwriting).
#
# We sub-sample to 3,000 rows so the whole audit pipeline runs in well under three minutes.

# %%
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame.sample(n=3000, random_state=42).reset_index(drop=True)

y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

print(f"Shape: {X.shape}")
print(f"Positive rate (>50K): {y.mean():.3f}")
print(f"Sensitive attributes available: sex, race, age")
print(f"Dtypes: {X.dtypes.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. The single production candidate: `skrub` + `TabICL`
#
# One pipeline. No tuning. The artifact pack is generated *for this exact estimator*.

# %%
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer
from tabicl import TabICLClassifier

pipe = make_pipeline(
    TableVectorizer(),
    TabICLClassifier(device="cpu", random_state=42),
)
pipe.fit(X_train, y_train)
print("Pipeline fitted.")

# %%
import skore

report = skore.EstimatorReport(
    pipe,
    fit=False,
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    pos_label=1,
)

metrics_df = report.metrics.summarize().frame()
display(metrics_df)

report.metrics.roc()
plt.show()

report.metrics.precision_recall()
plt.show()

# %% [markdown]
# ## 3. Subgroup performance — Annex IV §3
#
# Per-group AUC and accuracy on the test set. This is what §3 calls *"performance characteristics for specific persons or groups of persons on which the system is intended to be used."*

# %%
from sklearn.metrics import roc_auc_score, accuracy_score

y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

rows = []

# Per sex
for sex_value in sorted(X_test["sex"].dropna().unique()):
    mask = (X_test["sex"] == sex_value).values
    if mask.sum() < 10 or len(np.unique(y_test.values[mask])) < 2:
        continue
    rows.append({
        "group": f"sex = {sex_value}",
        "n": int(mask.sum()),
        "auc": roc_auc_score(y_test.values[mask], y_proba[mask]),
        "accuracy": accuracy_score(y_test.values[mask], y_pred[mask]),
    })

# Per age bin
age_bins = [(0, 30, "age < 30"), (30, 50, "age 30–50"), (50, 200, "age 50+")]
for lo, hi, label in age_bins:
    mask = ((X_test["age"] >= lo) & (X_test["age"] < hi)).values
    if mask.sum() < 10 or len(np.unique(y_test.values[mask])) < 2:
        continue
    rows.append({
        "group": label,
        "n": int(mask.sum()),
        "auc": roc_auc_score(y_test.values[mask], y_proba[mask]),
        "accuracy": accuracy_score(y_test.values[mask], y_pred[mask]),
    })

subgroup_df = pd.DataFrame(rows).round(4)
subgroup_df

# %% [markdown]
# ## 4. Render the Annex IV draft to PDF
#
# Skore 0.16 has no native PDF export, so we assemble an HTML dossier (Skore's own `_repr_html_` for the report, plus our subgroup table and risk/monitoring placeholder) and render it via headless Chromium.
#
# The output is the file your auditor will actually open.

# %%
import asyncio
from playwright.async_api import async_playwright

out_dir = Path("../recordings/final").resolve()
out_dir.mkdir(parents=True, exist_ok=True)
html_path = out_dir / "demo6_annex_iv.html"
pdf_path = out_dir / "demo6_annex_iv.pdf"

dtypes_html = (
    pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
    .to_html(index=False, classes="meta")
)

sections = []
sections.append("<h1>Annex IV — Technical Documentation Draft</h1>")
sections.append(
    "<p style='color:#666'>Generated by the :probabl. open tabular stack "
    "(skrub + TabICL + Skore). High-risk AI system under Article 6, EU AI Act.</p>"
)

sections.append("<h2>1. Intended purpose</h2>")
sections.append(
    "<p>Binary classifier predicting whether an individual's annual income exceeds "
    "USD 50,000, used here as a proxy for credit-screening / employment-shortlisting "
    "models that fall under Annex III §5 (creditworthiness) and §4 (employment) of the "
    "EU AI Act. Output is a probability score; downstream decision threshold is set by "
    "the deploying institution.</p>"
)

sections.append("<h2>2. Methods and dataset description</h2>")
sections.append(
    f"<p><b>Pipeline:</b> <code>skrub.TableVectorizer</code> → "
    f"<code>tabicl.TabICLClassifier(device='cpu', random_state=42)</code>. "
    f"No hyperparameter tuning was performed; TabICL is used at its pre-trained defaults.</p>"
)
sections.append(
    f"<p><b>Dataset:</b> UCI Adult (OpenML <code>adult</code> v2), sub-sampled to "
    f"<b>{df.shape[0]:,}</b> rows and <b>{df.shape[1] - 1}</b> features. "
    f"Stratified 80 / 20 split, <code>random_state=42</code>.</p>"
)
sections.append("<details><summary>Schema</summary>" + dtypes_html + "</details>")

sections.append("<h2>3. Performance and calibration (Skore EstimatorReport)</h2>")
sections.append(report._repr_html_())

sections.append("<h2>4. Subgroup performance</h2>")
sections.append(
    "<p>Per-group AUC and accuracy on the held-out test set. Required by Annex IV §3 "
    "(performance characteristics for specific groups).</p>"
)
sections.append(subgroup_df.to_html(index=False, classes="meta"))

sections.append("<h2>5. Risk management and monitoring</h2>")
sections.append(
    "<ul>"
    "<li><b>Drift monitoring:</b> input-distribution KS test against the training "
    "reference, weekly cadence; alert threshold p &lt; 0.01 on any feature.</li>"
    "<li><b>Performance re-evaluation:</b> scheduled quarterly back-test against "
    "freshly labelled production data; report regenerated by re-running this notebook.</li>"
    "<li><b>Human oversight:</b> probability scores are advisory only; final adverse "
    "decisions require reviewer sign-off, logged with the model version hash.</li>"
    "</ul>"
)

html = (
    "<html><head><meta charset='utf-8'><style>"
    "body{font-family:Inter,system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#1A1A2E}"
    "h1,h2{color:#1E22AA}"
    "h1{border-bottom:2px solid #1E22AA;padding-bottom:8px}"
    "table.meta{border-collapse:collapse;font-size:12px;margin:8px 0}"
    "table.meta td,table.meta th{border:1px solid #ddd;padding:4px 10px;text-align:left}"
    "code{background:#F2F2F8;padding:1px 4px;border-radius:3px}"
    "details{margin:8px 0}"
    "</style></head><body>"
    + "\n".join(sections)
    + "</body></html>"
)

html_path.write_text(html, encoding="utf-8")

async def render_pdf():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{html_path}")
        await page.pdf(path=str(pdf_path), format="A4", print_background=True)
        await browser.close()

await render_pdf()

print(f"PDF: {pdf_path}")
print(f"size: {pdf_path.stat().st_size / 1024:.1f} KB")

# %% [markdown]
# ## 5. Push the report to Skore Hub
#
# The PDF is the deliverable; the Hub link is the **living** version your CRO bookmarks.

# %%
project = skore.Project("debray.yann/demo6-audit-ready", mode="hub")
project.put("annex-iv-draft", report)
print(project)
print(f"https://skore.probabl.ai/{project.name}")

