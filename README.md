# Evaluate TabICL in Skore ⚖️

> *out-of-the-box → handles dirt → calibrated → distributional → small-data robust → audit-ready*

Five demos benchmarking [TabICL](https://github.com/soda-inria/tabicl) (the open Tabular Foundation Model from soda-inria) against scikit-learn `HistGradientBoosting`, evaluated end-to-end with [Skore](https://github.com/probabl-ai/skore) and pushed to Skore Hub.

## The five demos

| # | Notebook | Theme |
|---|---|---|
| 1 | `demos/demo1_five_minute_model.ipynb` | TabICL default vs HGBT default vs HGBT + Optuna |
| 2 | `demos/demo2_dirty_data.ipynb` | skrub `TableVectorizer` recovers dropped columns; TabICL on top |
| 3 | `demos/demo3_calibrated_probs.ipynb` | reliability diagrams + ECE on imbalanced fraud data |
| 4 | `demos/demo4_quantile_regression.ipynb` | HGBT q-ensemble vs TabICL's native distribution |
| 5 | `demos/demo5_small_data.ipynb` | learning-curve sweep 50 → 500 rows |

## Quickstart

```bash
# Python 3.12 venv
uv pip install --python .venv/bin/python jupyterlab optuna ipykernel

# Set secrets
cp .env.example .env  # add SKORE_HUB_API_KEY

# Launch JupyterLab and open any demo
.venv/bin/jupyter lab demos/
```

Every notebook calls `skore.login()` and pushes individual `EstimatorReport`s to `debray.yann/demoN-...` on Skore Hub.

## Notes

- Six demos × ~2 min each, runnable on macOS CPU (no CUDA). The longest cell is ~30 s.
- TabICL classifier + regressor checkpoints (~few hundred MB) download from HuggingFace on first use.
