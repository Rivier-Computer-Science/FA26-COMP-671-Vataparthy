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

# 3. ADD CUSTOMER HISTORY USING ONLY STRICTLY EARLIER EVENTS
def add_history(events):
    # Build each purchase's customer history using only strictly earlier events.
    # Calculate earlier purchase counts, spending/quantity averages, 30-day
    # frequency, recency, and cancellation rate. Return purchases with these
    # features; cancellations update history but do not become scored purchases.

    # Each customer's state stores running totals and a queue of purchase dates.
    history, result = {}, []
    for date, group in events.groupby("date", sort=True):
        rows = group.to_dict("records")
        for row in rows:
            state = history.setdefault(row["customer"], {
                "count": 0, "value": 0, "quantity": 0, "cancellations": 0,
                "last_purchase": None, "recent": deque()})
            if row["is_cancel"]:
                continue  # Cancellations affect history but are not scored orders.
            # Count purchases in [current time - 30 days, current time).
            while state["recent"] and state["recent"][0] < date - pd.Timedelta(days=30):
                state["recent"].popleft()
            # Zero is a placeholder when there are no earlier purchases.
            # make_features() adds a flag so the model can recognize this case.
            count = state["count"]
            row["previous_orders"] = count
            row["mean_value"] = state["value"] / count if count else 0
            row["mean_quantity"] = state["quantity"] / count if count else 0
            row["purchases_30d"] = len(state["recent"])
            # Divide seconds by 86,400 to express the gap in fractional days.
            row["days_since_purchase"] = ((date - state["last_purchase"]).total_seconds()
                                          / 86400 if count else 0)
            # Event-based rate, not the fraction of purchased units returned.
            total_events = count + state["cancellations"]
            row["cancellation_rate"] = state["cancellations"] / total_events if total_events else 0
            result.append(row)
        # Update AFTER scoring the group: simultaneous events cannot see each other.
        for row in rows:
            state = history[row["customer"]]
            if row["is_cancel"]:
                state["cancellations"] += 1
            else:
                state["count"] += 1
                state["value"] += row["value"]
                state["quantity"] += row["quantity"]
                state["last_purchase"] = date
                state["recent"].append(date)
    return pd.DataFrame(result)


def make_features(data, use_history):
    # Convert prepared purchases into numerical model inputs, excluding IDs.
    # False produces 7 current-transaction features (Model A); True produces
    # 16 transaction-plus-history features (Model B). Return a feature table.

    # log1p means log(1 + value): compress large amounts while allowing zeros.
    features = np.log1p(data[["value", "quantity", "products"]]).copy()
    # Sine/cosine preserve the circular nature of time: midnight is near 23:59,
    # and Sunday is next to Monday. These are hour and weekday features.
    hour = data["date"].dt.hour + data["date"].dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * data["date"].dt.dayofweek / 7)
    features["day_cos"] = np.cos(2 * np.pi * data["date"].dt.dayofweek / 7)
    if use_history:
        # A new customer has no purchase baseline; the flag distinguishes this.
        known = data["previous_orders"].gt(0)
        features["has_history"] = known.astype(float)
        for column in ["previous_orders", "mean_value", "mean_quantity",
                       "purchases_30d", "days_since_purchase"]:
            features[column] = np.log1p(data[column])
        features["cancellation_rate"] = data["cancellation_rate"]
        # Compare the current purchase with that customer's historical averages.
        # Set these differences to zero when no earlier purchase baseline exists.
        features["spending_change"] = np.where(known, np.log1p(data["value"]) - np.log1p(data["mean_value"]), 0)
        features["quantity_change"] = np.where(known, np.log1p(data["quantity"]) - np.log1p(data["mean_quantity"]), 0)
    return features


# 4. CHRONOLOGICAL SPLITS AND A CONTROLLED SYNTHETIC TEST
def split_data(orders):
    # Divide time-ordered purchases into training, validation, calibration,
    # and test tables. Training fits models; validation selects AE weights;
    # calibration sets anomaly thresholds; testing measures final performance.

    # Cut at 60%, 70%, and 80% of DISTINCT timestamps, not of invoice rows.
    # This keeps simultaneous purchases together, so row proportions can differ.
    times = orders["date"].drop_duplicates().to_numpy()
    if len(times) < 40:
        raise ValueError("Need at least 40 distinct timestamps.")
    a, b, c = [times[int(len(times) * part)] for part in [.60, .70, .80]]
    date = orders["date"]
    parts = [orders[date < a], orders[(date >= a) & (date < b)],
             orders[(date >= b) & (date < c)], orders[date >= c]]
    for name, part in zip(["Training", "Validation", "Calibration", "Test"], parts):
        print(f"{name}: {len(part):,} orders, {part.date.min()} to {part.date.max()}")
    return parts


def make_benchmark(test):
    # Create a labeled synthetic benchmark from held-out purchases without
    # modifying the original test table. Return originals plus altered copies,
    # with labels 0 for assumed negatives and 1 for injected anomalies.

    rng = np.random.default_rng(123)  # Identical benchmark for every model/seed.
    original = test.copy()
    original["kind"] = "original_unlabeled"
    # Add altered copies numbering about 15% of the original test orders.
    fake = test.sample(n=max(3, int(len(test) * .15)), random_state=123).copy()
    kinds = np.resize(["bulk", "high_value", "customer_shift"], len(fake))
    for index, kind in zip(fake.index, kinds):
        factor = rng.uniform(4, 10)
        # Bulk: increase both quantity and value by the same whole-number factor.
        if kind == "bulk":
            factor = int(np.ceil(factor))
            fake.loc[index, ["value", "quantity"]] *= factor
        # Spending shift: set value to 4-10 times the customer's earlier mean.
        elif kind == "customer_shift" and fake.loc[index, "previous_orders"] >= 3:
            fake.loc[index, "value"] = fake.loc[index, "mean_value"] * factor
        else:
            # Also use a high-value anomaly when customer history is too short.
            kind = "high_value"
            fake.loc[index, "value"] *= factor
        fake.loc[index, "kind"] = kind
    # Each fake is an independent next-order scenario with its original history.
    benchmark = pd.concat([original, fake], ignore_index=True)
    # Label 0 is an evaluation assumption, not confirmation that an order is normal.
    labels = np.r_[np.zeros(len(original), dtype=int), np.ones(len(fake), dtype=int)]
    return benchmark, labels


if __name__ == "__main__":
    main()
