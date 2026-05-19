# %% [markdown]
# # Demo 2 — Dirty Data, Clean Prediction
#
# ### Chapter 2 of the open tabular stack: *handles dirt*
#
# > **Real data is messy. Tools should hide the mess.**
# > *This is what Gaël Varoquaux's team has been building since dirty_cat, now skrub.*
#
# Three pipelines on the **employee salaries** dataset — strings, dates, divisions, job titles:
#
# | Pipeline | What it represents |
# |---|---|
# | `HGBT manual` | Drop everything that isn't a number. The do-it-yourself baseline. |
# | `TableVectorizer + HGBT` | Skrub recovers the columns you'd otherwise throw away. |
# | `TableVectorizer + TabICL` | The same vectorizer, now feeding a foundation model. |
#
# Skore decomposes the lift: how much comes from **preprocessing**, how much from the **model**.
#
# ---
#
# **Stack used:**
# - `skrub` — `TableVectorizer`, `fetch_employee_salaries`
# - `scikit-learn` — `HistGradientBoostingRegressor`, train/test split
# - `tabicl` — `TabICLRegressor`, the foundation model
# - `skore` — `EstimatorReport` + `ComparisonReport` + Hub

# %% 
my_workspace = "" 
# Set `my_workspace` to your Skore workspace name to upload the results at the end.
# If left empty, the project will be created in "local" mode and the results won't be
# uploaded to the Hub, but you can still see them in your local instance of Skore.
# %%
import os
from pathlib import Path

# %% [markdown]
# Load SKORE_HUB_API_KEY from .env at the repo root.
# _repo_root = Path.cwd()
# while not (_repo_root / ".env").exists() and _repo_root != _repo_root.parent:
#     _repo_root = _repo_root.parent
# _env_path = _repo_root / ".env"
# if _env_path.exists():
#     for _line in _env_path.read_text().splitlines():
#         _line = _line.strip()
#         if _line and not _line.startswith("#") and "=" in _line:
#             _k, _v = _line.split("=", 1)
#             os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
# %%
import skore
# if "SKORE_HUB_API_KEY" in os.environ:
skore.login()


# %% [markdown]
# ## 0. Setup

# %%
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt

BLUE = "#1E22AA"
ORANGE = "#F68D2E"
GRAY = "#94A3B8"
DARK = "#1A1A2E"

@contextmanager
def stopwatch(label):
    t0 = time.perf_counter()
    yield (lambda: time.perf_counter() - t0)
    print(f"  [{label}] wall-clock: {time.perf_counter() - t0:.1f}s")


# %% [markdown]
# ## 1. Employee salaries — strings, dates, divisions
#
# Public Montgomery County salary data: ~9k rows of mixed-type messiness. Free-text
# job titles, department codes, hire dates as strings, one lone numeric column.
#
# We sub-sample to **2,000 rows** so the notebook stays under three minutes wall-clock
# (TabICL inference is the floor).

# %%
from skrub.datasets import fetch_employee_salaries
from sklearn.model_selection import train_test_split

data = fetch_employee_salaries()
X = data.X[:4000]
y = data.y[:4000]
# only using part of the dataset to keep the memory under ~ 10 Gb with TabICL,
# and runtime under a couple of minutes for each cell.
# If you experience kernel crashes, try reducing this further.

# %%
from skrub import TableReport
TableReport(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %% [markdown]
# ## 2. Three pipelines
#
# 1. **`HGBT manual`** — keep only `year_first_hired`. Everything else gets dropped
#    because HGBT doesn't natively eat strings. This is the "I haven't met skrub yet" baseline.
# 2. **`TableVectorizer + HGBT`** — one line, all columns recovered.
# 3. **`TableVectorizer + TabICL`** — same vectorizer, foundation model on top.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from skrub import TableVectorizer
from tabicl import TabICLRegressor

numeric_only = X.select_dtypes(include="number").columns.tolist()

manual_preproc = ColumnTransformer(
    [("keep_numeric", "passthrough", numeric_only)],
    remainder="drop",
)

pipelines = {
    "HGBT manual": make_pipeline(
        manual_preproc, HistGradientBoostingRegressor(random_state=42)
    ),
    "TableVectorizer + HGBT": make_pipeline(
        TableVectorizer(), HistGradientBoostingRegressor(random_state=42)
    ),
    "TableVectorizer + TabICL": make_pipeline(
        TableVectorizer(), TabICLRegressor(device="cpu", random_state=42, batch_size=2)
    ),
}

fitted = {}
for name, pipe in pipelines.items():
    with stopwatch(name):
        pipe.fit(X_train, y_train)
    fitted[name] = pipe

# %% [markdown]
# ## 3. Skore — one comparison, three models
#
# Each fitted pipeline becomes an `EstimatorReport`. The `ComparisonReport` then
# renders the metrics table and the prediction-error scatter side-by-side.

# %%
import skore

reports = {}
for name, pipe in fitted.items():
    print(f"creating report for {name}")
    reports[name] = skore.EstimatorReport(
            pipe, fit=False,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )

comparison = skore.ComparisonReport(reports=reports)
metrics = comparison.metrics.summarize()
metrics.frame()

# %%
prediction_display = comparison.metrics.prediction_error()
prediction_display.plot()

# %% [markdown]
# ## 4. The chart — R² across the three pipelines
#
# Gray = numeric-only. Blue = skrub recovers the columns. Orange = skrub feeds TabICL.

# %%
results = {}
for name, report in reports.items():
    results[name] = {
        "r2": report.metrics.r2(),
        "rmse": report.metrics.rmse(),
    }

colors = {
    "HGBT manual": GRAY,
    "TableVectorizer + HGBT": BLUE,
    "TableVectorizer + TabICL": ORANGE,
}

names = list(results.keys())
r2s = [results[n]["r2"] for n in names]
bar_colors = [colors[n] for n in names]

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.barh(names, r2s, color=bar_colors, edgecolor="white", linewidth=2, zorder=3)
for bar, n in zip(bars, names):
    w = bar.get_width()
    ax.text(
        w + 0.01, bar.get_y() + bar.get_height() / 2,
        f"R²={results[n]['r2']:.3f}  ·  RMSE={results[n]['rmse']:,.0f}",
        va="center", ha="left", fontsize=10, color=DARK,
    )

ax.set_xlim(0, max(r2s) * 1.35)
ax.invert_yaxis()
ax.set_xlabel("Test R²", color=DARK, fontsize=11)
ax.set_title(
    "Skrub recovers the columns. TabICL squeezes them dry.",
    color=BLUE, fontsize=14, fontweight="bold", loc="left", pad=18,
)
ax.grid(True, axis="x", color="#E0E0E0", linestyle="--", linewidth=0.6, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax.spines[spine].set_color("#999999")

fig.text(
    0.99, 0.02, ":probabl.  ·  by the scikit-learn founders",
    ha="right", color="#666666", fontsize=9, style="italic",
)

plt.tight_layout()
import os
os.makedirs("../recordings/", exist_ok=True)
plt.savefig("../recordings/demo2_chart.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.show()

# %% [markdown]
# ## 5. Push to Skore Hub
#
# The comparison object is the artifact. One `put` and the URL is shareable.

# %%
if my_workspace:
    mode = "hub"
else:
    mode = "local"

project = skore.Project(f"{my_workspace}/demo2-dirty-data", mode=mode)
for name, rep in reports.items():
    project.put(name, rep)

if mode == "hub":
    print(f"https://skore.probabl.ai/{project.name}")


