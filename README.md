# Evaluate TabICL in Skore ⚖️

> *out-of-the-box → handles dirt → calibrated → distributional → small-data robust → audit-ready*

Five demos benchmarking [TabICL](https://github.com/soda-inria/tabicl) (the open Tabular Foundation Model from soda-inria) against scikit-learn `HistGradientBoosting`, evaluated end-to-end with [Skore](https://github.com/probabl-ai/skore) and pushed to Skore Hub.

## The five demos

| # | Notebook | Theme |
|---|---|---|
| 1 | `demo1_five_minute_model.py` | TabICL default vs HGBT default vs HGBT + Optuna |
| 2 | `demo2_dirty_data.py` | skrub `TableVectorizer` recovers dropped columns; TabICL on top |
| 3 | `demo3_calibrated_probs.py` | reliability diagrams + ECE on imbalanced fraud data |
| 4 | `demo4_quantile_regression.py` | HGBT q-ensemble vs TabICL's native distribution |
| 5 | `demo5_small_data.py` | learning-curve sweep 50 → 500 rows |

## Quickstart

```bash
pip install -r requirements.txt
```

1. create an account at [skore login](skore.probabl.ai/login?register=true)
2. 

## Notes

- Six demos of ~2 min each, runnable on macOS CPU (no CUDA). The longest cell is ~30 s.
- TabICL classifier + regressor checkpoints (~few hundred MB) download from HuggingFace on first use.
