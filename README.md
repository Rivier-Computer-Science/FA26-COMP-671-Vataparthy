# ShopSense

**Deep Learning-Based Detection of Unusual Customer Purchasing Behavior in Online Retail**

**Author:** Angel Akhila Vataparthy  
**Course:** FA26 - COMP-671-AO1 - DEEP LEARNING  
**Institution:** Rivier University

## Overview

ShopSense is an offline Python project that identifies unusual retail purchases. It investigates whether adding a customer's earlier purchasing history improves an autoencoder's anomaly detection compared with using only the current order.

The project includes data preparation, model training, validation, threshold calibration, held-out testing, and an optional prediction demonstration. All application code is in `main.py`. The interface is the terminal, and the models run on the CPU.

An unusual score suggests that an order deserves review. It does not establish fraud.

## Project structure

| File | Purpose |
| --- | --- |
| `main.py` | Complete preprocessing, training, evaluation, and prediction program |
| `README.md` | Project overview, installation, and run instructions |
| `requirements.txt` | Direct Python dependencies for initial installation |
| `REPRODUCIBILITY.md` | Exact experiment settings, environment capture, data verification, and retraining instructions |
| `Online_Retail.xlsx` | Dataset workbook placed beside `main.py` |

The reproduction instructions also generate `requirements-lock.txt`, `environment.txt`, `checksums.txt`, `source_commit.txt`, and `results.txt`. These files record the environment and output of an actual run. They must be generated on the machine used for that run.

## Dataset

The project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), containing 541,909 product lines from a UK retailer, dated December 2010 to December 2011. Product lines are grouped into invoices before modeling.

1. Download the dataset from the UCI page and extract the Excel workbook if the download is an archive.
2. Place the workbook in the same folder as `main.py`.
3. Name it `Online_Retail.xlsx` or `Online Retail.xlsx`. Keep only one copy under these accepted names. The code checks the filename with a space first.

A filename such as `Online_Retail(1).xlsx` must be renamed. The workbook must retain the original column names. The program reads `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, and `CustomerID`; description and country are not model inputs.

The loader removes duplicate lines and excludes incomplete or invalid invoices. Valid purchases become the orders to score. Cancellation invoices contribute to historical features. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the workbook checksum and expected preprocessing counts.

## Dependencies

- Python and pip. The setup examples below use 64-bit Python 3.12. For a recorded run, use the Python version in its `environment.txt`.
- NumPy, pandas, openpyxl, scikit-learn, and PyTorch.
- A standard CPU. No GPU or cloud service is required.

Internet access is needed to obtain the dataset and install packages. Training, evaluation, and prediction then work offline.

`requirements.txt` lists the necessary packages without inventing the versions used for the original results. For a reproducible submission, capture the installed versions from a successful run using the commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Use that captured `requirements-lock.txt` when recreating the run.

## Setup

### Windows PowerShell

Open a terminal in the repository folder. If the project already runs successfully, capture that environment before creating or changing environments.

For a fresh setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

These commands use the environment's Python directly, so PowerShell activation is unnecessary. If the `py` launcher is unavailable, create the environment with `python -m venv .venv` using your selected Python installation.

If a captured `requirements-lock.txt` is available, follow the recreation steps in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) instead of installing unpinned dependencies.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The program uses CPU tensors even when the installed PyTorch package supports other devices. For a platform-specific installation, use the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Run the project

Windows, using the environment created above:

```powershell
.\.venv\Scripts\python.exe -u main.py
```

In an activated environment on any platform:

```bash
python -u main.py
```

The program performs the following steps:

1. Reads and cleans the workbook, then aggregates product lines into invoices.
2. Builds customer features using strictly earlier events.
3. Creates chronological training, validation, calibration, and test partitions.
4. Fits preprocessing on training data and trains the comparison models.
5. Selects autoencoder weights using validation reconstruction error.
6. Sets alert thresholds using the calibration partition.
7. Evaluates every model on the same held-out synthetic benchmark.
8. Prints detailed results, averages across seeds, and original orders for review.
9. Offers one optional manually entered order for prediction.

Loading the Excel workbook happens before the first progress message and can take time. Runtime depends on the computer and installed packages.

### Running in Visual Studio Code

Open the repository folder, press **Ctrl + Shift + P**, select **Python: Select Interpreter**, and choose the environment containing the dependencies. Open `main.py` and run it with **Ctrl + F5**, or use the terminal commands above.

If `ModuleNotFoundError: No module named 'torch'` appears, check that VS Code is using the same interpreter in which the packages were installed. The explicit `.venv` commands above help avoid this mismatch.

## Models and controlled comparison

| Method | How it scores unusual orders |
| --- | --- |
| Autoencoder | Mean squared reconstruction error from a small neural network |
| Isolation Forest | Reversed Isolation Forest score, so a larger value means more unusual |
| PCA | Mean squared error after compression and reconstruction |

Each method uses two feature sets:

- **Transaction features, Model A for the autoencoder:** 7 inputs describing value, total quantity, distinct products, and cyclic purchase hour and weekday.
- **Transaction plus history, Model B for the autoencoder:** 16 inputs, adding previous purchase count, mean spending and quantity, purchases in the previous 30 days, recency, cancellation rate, availability of history, and changes from earlier spending and quantity.

Models share the same data partitions and synthetic benchmark. Autoencoder and Isolation Forest use seeds `42`, `43`, and `44`. PCA uses a deterministic full solver and runs once per feature set. The autoencoder uses MSE loss, Adam, and backpropagation. Detailed settings and their locations in the code are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Evaluation and results

Splits occur at 60%, 70%, and 80% of distinct purchase timestamps. All orders at the same timestamp remain together, so the proportions of invoice rows may differ from those percentages.

The test benchmark combines original test orders with altered copies representing bulk purchases, unusually high values, and changes in customer spending. Original orders are assumed negatives because the dataset has no verified anomaly labels.

The program reports precision, recall, F1, ROC-AUC, average precision, original-order alert rate, and recall for each synthetic anomaly type. Recall is the synthetic anomaly detection rate. PCA and autoencoder outputs also include mean reconstruction error for original and synthetic orders.

The following values summarize the previously reported run:

| Features | Method | F1 mean | Average precision mean |
| --- | --- | ---: | ---: |
| Transaction | Autoencoder | 0.2393 | 0.4172 |
| Customer | Autoencoder | 0.2757 | 0.4647 |
| Transaction | Isolation Forest | 0.1112 | 0.2860 |
| Customer | Isolation Forest | 0.0714 | 0.2879 |
| Transaction | PCA | 0.2137 | 0.4305 |
| Customer | PCA | 0.2187 | 0.4520 |

Autoencoder and Isolation Forest entries average three seeds. Each PCA entry comes from one fit. These are reference results from the supplied program output, not guarantees for every software environment.

Customer history improved mean autoencoder F1 and average precision, but the improvement varied across seeds. The customer autoencoder detected only about 13% to 25% of synthetic anomalies at its calibrated threshold. PCA remained competitive and achieved higher mean ROC-AUC. The findings provide partial support for the proposed benefit of customer history.

## Optional prediction demonstration

After evaluation, press Enter at the customer prompt to finish without entering an order. Evaluation does not require manual input.

To demonstrate an intentionally extreme order, enter:

| Prompt | Input |
| --- | --- |
| Customer ID | `17850` |
| Order value in GBP | `1000000` |
| Total item quantity | `100000` |
| Number of distinct products | `5` |
| Order date and time | `2011-12-10 12:00` |

The date must be timezone-free and later than the last event in the loaded dataset. Value must be positive and finite. Quantity and product count must be positive whole numbers, with distinct products no greater than quantity.

The program uses the customer autoencoder trained with the first configured seed, its training scaler, and its calibrated threshold. It prints the score and either `Unusual: review this order.` or `Within the learned range.` The supplied extreme-order run produced a score of `7.8452` against a threshold of `0.6784`; compare the values from your actual run.

Scores are reconstruction errors, not probabilities. The entered order does not change training, evaluation results, or stored customer history. Only one manual order is accepted per execution.

## Configuration and retraining

Configuration is implemented directly in `main.py`; the program does not load a separate YAML, JSON, or environment configuration file and accepts no training command-line options. `REPRODUCIBILITY.md` documents the existing settings rather than changing how the program works.

Each execution retrains all comparison models. The program keeps models, scalers, and thresholds in memory and does not save or load trained weights. Reproduction therefore uses the documented retraining commands. Console output is saved only if you use the logging commands in the reproduction guide.

## Limitations and responsible use

- Synthetic benchmark scores do not measure verified real-world fraud detection. An original-order alert rate is not a confirmed false-positive rate.
- Large legitimate purchases, seasonal changes, and limited customer history can complicate interpretation.
- Fixed seeds improve repeatability, but exact results may differ across platforms and software versions. Repeating the same seeds does not create new independent experiments.
- PCA standard-deviation entries are `NaN` because a single result cannot provide sample standard deviation. Isolation Forest MSE entries are `NaN` because it does not reconstruct inputs.
- The application has no GUI, automatic transaction blocking, saved-model inference mode, or feature-level causal explanations.
- Customer IDs and purchase histories require protected storage and appropriate access controls. Any consequential decision requires human review.

## Dataset attribution and documentation

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

The dataset is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The program transforms the data during preprocessing; retain this attribution if the workbook or derived data is distributed. This dataset license does not automatically specify a license for the project code.

- [Reproduction instructions](REPRODUCIBILITY.md)
- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [Python virtual environments](https://docs.python.org/3/library/venv.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
