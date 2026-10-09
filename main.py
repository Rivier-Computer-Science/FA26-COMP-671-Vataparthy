"""ShopSense: a small, single-file CPU experiment and prediction demo.

Dataset: Chen, D. (2015), UCI Online Retail, doi.org/10.24432/C5BW33.
Scope: score positive purchases with known customers; keep cancellations as history.
Synthetic benchmark results are NOT verified real-world fraud accuracy.
"""

# 1. IMPORTS AND SETTINGS
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# Repeat the random models with three seeds to check variation between runs.
SEEDS = [42, 43, 44]
# Maximum training passes; early stopping may finish before this limit.
EPOCHS = 40
# Use two CPU threads; the models and tensors below stay on the CPU.
torch.set_num_threads(2)


# 2. CLEAN INVOICE LINES AND KEEP PURCHASES AND CANCELLATIONS
def load_events(path):
    # Read the Excel dataset, clean invoice lines, and combine each invoice's
    # product lines into one event. Return purchases and cancellations in time
    # order, keeping only complete, valid invoices with known customer IDs.

    # Read IDs as text because they identify records, not numerical quantities.
    raw = pd.read_excel(path, engine="openpyxl", dtype={
        "InvoiceNo": "string", "CustomerID": "string", "StockCode": "string"})
    # Invalid numeric/date text becomes missing so the validity check can reject it.
    data = raw.drop_duplicates().copy()
    for column in ["Quantity", "UnitPrice"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    # UCI marks cancelled invoices with a C prefix; their quantities are negative.
    data["is_cancel"] = data["InvoiceNo"].str.upper().str.startswith("C", na=False)
    # Require valid IDs/dates, whole nonzero quantities, and positive prices.
    # Purchases must have positive quantities; C-prefixed cancellations must be negative.
    valid = (data[["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate"]].notna().all(axis=1)
             & np.isfinite(data["Quantity"]) & np.isfinite(data["UnitPrice"])
             & data["Quantity"].ne(0) & data["Quantity"].mod(1).eq(0)
             & data["UnitPrice"].gt(0)
             & data["Quantity"].lt(0).eq(data["is_cancel"]))
    # Discard entire incomplete/invalid invoices, rather than partial baskets.
    bad_ids = data.loc[~valid, "InvoiceNo"]
    data = data.loc[valid & ~data["InvoiceNo"].isin(bad_ids)].copy()
    # One invoice must belong to one customer and have one recorded timestamp.
    conflicts = data.groupby("InvoiceNo")[["CustomerID", "InvoiceDate"]].nunique().gt(1).any(axis=1)
    data = data.loc[~data["InvoiceNo"].isin(conflicts[conflicts].index)].copy()
    # Store cancellation magnitudes as positive; is_cancel retains their meaning.
    data["Quantity"] = data["Quantity"].abs()
    data["amount"] = data["Quantity"] * data["UnitPrice"]
    # Sum line values/quantities; count distinct products to measure basket size.
    events = data.groupby("InvoiceNo").agg(
        customer=("CustomerID", "first"), date=("InvoiceDate", "first"),
        value=("amount", "sum"), quantity=("Quantity", "sum"),
        products=("StockCode", "nunique"), is_cancel=("is_cancel", "first")).reset_index()
    events = events.sort_values(["date", "InvoiceNo"]).reset_index(drop=True)
    purchases = (~events["is_cancel"]).sum()
    if purchases < 100 or not np.isfinite(events[["value", "quantity", "products"]]).all().all():
        raise ValueError("Need at least 100 valid orders with finite amounts.")
    print(f"Read {len(raw):,} lines; kept {purchases:,} purchases and {events.is_cancel.sum():,} cancellations.")
    return events

if __name__ == "__main__":
    main()
