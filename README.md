# ELEI: Evolutionary LLM-guided Equation Identifier

Code for [Evolutionary Identification of Scientific Equations Using
Language-Model-Guided Symbolic Regression](https://doi.org/10.1145/3795101.3805332)
(GECCO Companion 2026).

## Install

Validated with Python 3.9.25. Create an environment and install dependencies:

```bash
python -m venv .venv
# Activate: Windows PowerShell: .\.venv\Scripts\Activate.ps1
#           Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Qwen3-Max is the default. Set the API key in your shell (PowerShell):

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
python run_experiment.py
```

To use DeepSeek:

```powershell
$env:ELEI_PROVIDER = "deepseek"
$env:DEEPSEEK_API_KEY = "your-key"
python run_experiment.py
```

For Bash, use `export DASHSCOPE_API_KEY=your-key` and
`export ELEI_PROVIDER=deepseek` instead of PowerShell assignments.
API calls use your provider account. Model names and endpoints are in
`LLM_CONFIG` in `config.py`; live API aliases may change over time.

Select a dataset by editing `DATA_CONFIG` in `config.py`, for example:

```python
DATA_CONFIG = {
    "group": "Data/srsd-feynman_easy",
    "dataset": "feynman-i.12.1",
}
```

Edit `DATA_CONFIG_LIST` for a batch, then run `python run_batch.py`.
The supplied batch list contains the 50 Hard tasks. Available data groups are
`srsd-feynman_easy`, `srsd-feynman_medium`, `srsd-feynman_hard` and `sr-bench`,
all under `Data/`.

Each dataset contains `train_points.npy` and `test_points.npy`: numeric arrays
with input variables in the first columns and the target in the last column.
The corresponding `*-info.csv` provides variable descriptions. `eq/` contains
reference SRSD equations for inspection; these are not passed to the LLM.

## Implementation

1. **Preprocessing:** scale extreme numerical ranges, read variable descriptions,
   and choose 30 representative samples using K-Means.
2. **Generation:** send descriptions, ranges, samples, elite expressions and
   semantic memory to the LLM to propose 5 expression skeletons. Qwen and
   DeepSeek use the Chinese prompt and parser.
3. **Coefficient fitting:** fit coefficients with SciPy `curve_fit` (LM, with
   TRF fallback), using up to 10 random initializations. Prune coefficients
   when the fitting error changes negligibly.
4. **Selection and memory:** deduplicate using symbolic structure signatures,
   then retain up to 10 elites ordered by training R² and RMSE. Feed the elites
   and updated semantic summary into the next round.
5. **Termination and reporting:** stop when training R² >= 0.999 or after 9
   refinement rounds following initialization. Evaluate final candidates on
   train/test data and order the displayed results by the composite score
   below (lower is better):

   `score = 2 * (1 - R²) + 0.5 * log(1 + operator_count)`

The score uses R² clipped to [-1, 1]. At equal R², fewer operators rank first.
Search-time elite selection uses R²/RMSE; final display ordering uses test score.
Numerical R² and symbolic equivalence are separate evaluation criteria.

Outputs are saved under
`outputs/<provider>/<group>/<dataset>/<timestamp>/`: the data summary,
representative samples, search state and final train/test metrics.
Each invocation starts a new run.

## Files and checks

`main_loop.py` coordinates the search. `modules/` contains preprocessing,
prompts, model calls, fitting, structure handling, scaling and state storage.
`config.py` controls datasets, backends and iteration settings.

Run the offline checks with:

```bash
python -m unittest discover -s tests -v
```

These exercise preprocessing, fitting, evaluation and output handling using
synthetic data and mocked model replies, without API calls.
