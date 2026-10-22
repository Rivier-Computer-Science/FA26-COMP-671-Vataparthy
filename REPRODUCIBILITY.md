# ShopSense configuration and reproducibility

This guide documents the final single-file `main.py` implementation and explains how to record and repeat a complete run. All application logic remains in `main.py`.

## Reproduction status

The program output records the seeds, data counts, and performance results. It does not record the original Python or package versions. Therefore `requirements.txt` contains the direct dependencies without guessed version numbers. It is sufficient as an initial dependency list, but it does not fully specify the environment of the reported results.

Before final submission, generate the environment snapshot and run log below using the interpreter in which the project actually runs. If that environment has changed since the reported experiment, record a fresh run and label its results accordingly. Do not claim that newly captured versions describe an older run unless they are known to be the same.

## Experiment configuration

These values are already implemented in the code. This Markdown file is documentation and is not read by the program. Preserve the values when reproducing the reference experiment.

| Setting | Value | Location in `main.py` |
| --- | --- | --- |
| Training seeds | `42, 43, 44` | `SEEDS` |
| Maximum epochs | `40` | `EPOCHS` |
| Device and PyTorch threads | CPU, 2 threads | Imports and settings |
| Transaction feature count | 7 | `make_features()` |
| Customer feature count | 16 | `make_features()` |
| Historical frequency window | Previous 30 days | `add_history()` |
| Split boundaries | 60%, 70%, 80% of distinct purchase timestamps | `split_data()` |
| Input scaling | `StandardScaler`, fitted only on training features | `run_experiment()` |
| Model input dtype | `float32` | `run_experiment()` |
| Autoencoder layer sizes | Input, 16, 4, 16, output | `train_autoencoder()` |
| Activation | ReLU in hidden layers, linear output | `train_autoencoder()` |
| Loss | Mean squared error | `train_autoencoder()` |
| Optimizer | Adam | `train_autoencoder()` |
| Learning rate | `0.001` | `train_autoencoder()` |
| Batch size | Up to 256 | `train_autoencoder()` |
| Early-stopping patience | 10 consecutive epochs without sufficient improvement | `train_autoencoder()` |
| Minimum validation improvement | More than `0.00001` MSE | `train_autoencoder()` |
| Checkpoint selection | Restore the best accepted validation weights | `train_autoencoder()` |
| Isolation Forest trees | 100 | `run_experiment()` |
| Isolation Forest parallel jobs | 2 | `run_experiment()` |
| PCA | 4 components, `svd_solver="full"` | `run_experiment()` |
| PCA repetition | Once per feature set | `run_experiment()` |
| Synthetic sample seed | `123` for pandas sampling and the NumPy generator | `make_benchmark()` |
| Number of synthetic copies | `max(3, int(test_orders * 0.15))` | `make_benchmark()` |
| Anomaly threshold | 98th percentile of calibration scores | `run_experiment()` |
| Alert decision | Score strictly greater than threshold | `run_experiment()` and `predict_order()` |
| Demo model | Customer autoencoder with the first configured seed, 42 | `run_experiment()` |

The final code uses patience **10**, even though earlier project discussions considered 5. All six reported autoencoder fits reached the 40-epoch limit. The code restores the best validation checkpoint rather than automatically keeping the final epoch.

### Synthetic anomalies

- **Bulk:** multiply value and quantity by the same integer factor from 5 through 10, produced by taking the ceiling of a uniform draw between 4 and 10.
- **High value:** multiply order value by a uniform factor between 4 and 10.
- **Customer spending change:** set value to the customer's earlier mean spending multiplied by a uniform factor between 4 and 10. This requires at least three earlier purchases; otherwise the generator uses a high-value anomaly.

Synthetic orders are independent copies with the original order's preceding history. They do not alter the histories of other benchmark rows. The original test orders receive label 0 by assumption; injected copies receive label 1.

### Separation of data roles

Historical features use strictly earlier events. Orders sharing a timestamp cannot use one another as history. Earlier original test events can update the history of later test orders, while model weights, scalers, and thresholds remain fixed.

Training fits models and preprocessing. Validation selects neural-network weights. Calibration determines thresholds. Final testing measures performance on a separate chronological period. Both feature settings use the same splits, synthetic cases, and training procedure. Adding input features also changes the network parameter count, so the comparison does not hold parameter count constant.

## Data verification

Use the original workbook from the [UCI Online Retail page](https://archive.ics.uci.edu/dataset/352/online+retail). Place one accepted filename beside `main.py`: `Online_Retail.xlsx` or `Online Retail.xlsx`.

The SHA-256 of the workbook supplied with this project is:

```text
43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d
```

This checksum identifies the supplied file, not a checksum published by UCI. Renaming the workbook preserves it. Opening and re-saving the workbook can change the file bytes even when visible cells look unchanged.

Expected counts from that workbook and the final preprocessing code:

| Item | Count |
| --- | ---: |
| Original product lines | 541,909 |
| Retained purchase invoices | 18,472 |
| Retained cancellation invoices | 3,653 |
| Training purchases | 10,967 |
| Validation purchases | 1,858 |
| Calibration purchases | 1,856 |
| Test purchases | 3,791 |
| Synthetic test copies | 568 |
| Combined benchmark rows | 4,359 |

The held-out test period begins on October 26, 2011 and ends on December 9, 2011. If counts differ, check the workbook and preprocessing before interpreting differences in model scores.

## Record a run on Windows

Use PowerShell in the repository folder. Select the correct interpreter in VS Code first. The commands below use `python` to mean the interpreter that runs the project successfully. If necessary, replace it with `.\.venv\Scripts\python.exe` or the full path to that interpreter.

### 1. Confirm the interpreter and capture the environment

```powershell
python -c "import sys; print(sys.executable)"
python -m pip check
python -m pip freeze | Out-File -Encoding utf8 requirements-lock.txt
python -c "import platform, sys; print('Python:', sys.version); print('Platform:', platform.platform()); print('Architecture:', platform.machine()); print('Processor:', platform.processor())" | Out-File -Encoding utf8 environment.txt
python -m pip --version | Out-File -Encoding utf8 -Append environment.txt
```

`requirements-lock.txt` captures installed versions, including indirect dependencies. Despite the filename, `pip freeze` creates an installed-package snapshot rather than a complete cross-platform lockfile. Keep the same Python version, operating system, and architecture when recreating the environment. A dedicated project environment avoids unrelated packages and local editable installations in this snapshot.

### 2. Record source and data identity

Commit the final `main.py` and configuration changes before recording the source commit. Check that `main.py` has no uncommitted changes:

```powershell
git diff --exit-code HEAD -- main.py
git rev-parse HEAD | Out-File -Encoding utf8 source_commit.txt
Get-FileHash -Algorithm SHA256 main.py, Online_Retail.xlsx | Format-List | Out-File -Encoding utf8 checksums.txt
```

If the diff command shows changes, commit or deliberately document them before attributing results to that commit. If the workbook is named `Online Retail.xlsx`, replace `Online_Retail.xlsx` in the hash command with the quoted name `"Online Retail.xlsx"`.

### 3. Retrain and save evaluation output

```powershell
"" | python -u main.py 2>&1 | Tee-Object -FilePath results.txt
if ($LASTEXITCODE -ne 0) { throw "The run failed. Check results.txt." }
```

The blank input skips the optional prediction prompt after evaluation. The command still runs preprocessing, training, validation, calibration, and final testing. Confirm that `results.txt` contains all model rows, the seed summary, and the highest-scoring original orders, without a traceback.

For an interactive prediction demonstration, run `python -u main.py` without the pipe and use the example in the README. This starts a new training run. Alternatively, leave an interactive run open at the customer prompt and demonstrate prediction in that same session.

The listed capture commands overwrite files of the same name. Preserve earlier experiments separately before recording another configuration.

## Equivalent capture on macOS or Linux

In the activated project environment, use:

```bash
python -m pip check
python -m pip freeze > requirements-lock.txt
python -c "import platform, sys; print('Python:', sys.version); print('Platform:', platform.platform()); print('Architecture:', platform.machine()); print('Processor:', platform.processor())" > environment.txt
python -m pip --version >> environment.txt
git diff --exit-code HEAD -- main.py
git rev-parse HEAD > source_commit.txt
python -c "import hashlib; from pathlib import Path; names=['main.py', 'Online_Retail.xlsx']; print('\n'.join(hashlib.sha256(Path(n).read_bytes()).hexdigest()+'  '+n for n in names))" > checksums.txt
python -u main.py </dev/null > results.txt 2>&1
```

EOF skips the optional demo. Check the command's exit status and inspect the complete results log. Adjust the quoted dataset filename in the checksum command if needed.

## Recreate a captured environment and retrain

Use the source revision recorded in `source_commit.txt`, the verified workbook, and the Python version recorded in `environment.txt`. In a fresh environment on the same platform, install the captured packages:

```bash
python -m pip install -r requirements-lock.txt
python -m pip check
python -u main.py
```

For a Windows or Linux snapshot containing a PyTorch CPU build such as `torch==...+cpu`, make the official CPU wheel index available during installation:

```bash
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

No pretrained weights are required because this implementation retrains on each execution. It does not save model checkpoints, scalers, or thresholds to disk and has no separate prediction-only command. Use the exact retraining procedure above to reconstruct them in memory.

Fixed seeds reduce variation but do not guarantee identical numerical results across library releases, hardware, or operating systems. Compare data counts and experiment settings first, then compare the new metrics with the recorded log. Repeating the same fixed-seed run does not add independent experimental evidence.

## Materials to include in the repository

Commit `main.py`, `README.md`, `requirements.txt`, and `REPRODUCIBILITY.md`. After capturing a successful run, also commit `requirements-lock.txt`, `environment.txt`, `source_commit.txt`, `checksums.txt`, and `results.txt`. The data may be supplied in the repository with attribution or obtained through the download instructions.

Exclude `.venv/` and `__pycache__/` from Git. Keep actual project commits visible in the assigned repository. The report PDF and presentation/video have their own Canvas submission destinations in the assignment instructions.

## Supporting documentation

- [pip freeze](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)
