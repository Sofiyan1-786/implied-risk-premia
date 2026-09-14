# Learning the Inverse of Markowitz: reproducibility package

This repository reproduces every number, table and figure of the paper
*Learning the Inverse of Markowitz: Decision-Aware Implied Returns under Shorting Constraints*.
All data are synthetic and generated from fixed random seeds, so a full run regenerates the
results bit for bit.

## What you need (2 files, nothing else)

| File | Content |
|---|---|
| `mmlab.py` | The library: simulators, mean-variance and CRRA optimisers (with active-set polish), KKT inverse, learned surrogate, evaluation battery, premium data-generating processes and models |
| `unified_implied_returns_and_risk_premia.ipynb` | The notebook (already embeds a copy of `mmlab.py`, so running it alone regenerates everything) |

Everything else in this folder (`tests/`, `build_notebook.py`, `run_notebook.py`,
`outputs/`, `paper/`, `docs/`, precomputed CSVs/figures/HTML) is optional
convenience material. You do not need any of it to reproduce the results.

## Requirements

* Python 3.11 or newer (developed with 3.14)
* `pip install numpy scipy pandas matplotlib scikit-learn statsmodels cvxpy clarabel scs ipykernel nbformat nbclient nbconvert`
  (or `pip install -r requirements.txt` if you keep that file)
* About 2 GB of RAM and 12 CPU cores for the full run (fewer cores work, the run takes longer)
* No proprietary software: the conic solvers CLARABEL and SCS install via `pip`

## Reproduce

```bash
# 1. (optional) isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install numpy scipy pandas matplotlib scikit-learn statsmodels cvxpy clarabel scs ipykernel nbformat nbclient nbconvert

# 2. run the notebook (pick one)
jupyter nbconvert --to notebook --execute unified_implied_returns_and_risk_premia.ipynb --output unified_implied_returns_and_risk_premia.ipynb --ExecutePreprocessor.timeout=-1
# or simply open it in Jupyter / VS Code and Run All
```

For a quick smoke test set `MMLAB_FAST=1` before executing (a few minutes, small samples;
its numbers are not those of the paper). The full run takes about 40 minutes on 12 cores
and rewrites the executed notebook plus `outputs/table_*.csv`, `outputs/figNN_*.png|pdf`,
`outputs/paper_figures/FigN.*`, and `outputs/headline_results.json`.

## Reproducibility contract

* Training seeds (0 to 119) and evaluation seeds (1000 and above) never overlap.
* All random forests use `random_state=42`; parallel execution does not change the results.
* The notebook contains a copy of `mmlab.py`. For reuse, `import mmlab` instead of
  copy-pasting notebook cells.
* `MMLAB_FAST=1` is for smoke testing only; its numbers are not those of the paper.

## Licence

MIT, see `LICENSE`.

## Citation

See `CITATION.cff`.
