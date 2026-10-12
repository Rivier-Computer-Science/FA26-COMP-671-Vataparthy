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


# 5. MODEL SCORES AND AUTOENCODER TRAINING
def anomaly_scores(model, values):
    # Score already-scaled rows using a fitted model; larger means more unusual.
    # Isolation Forest uses its reversed native score. PCA and the autoencoder
    # use mean squared reconstruction error across features. Return one score
    # per row; these scores are not probabilities or directly comparable across models.

    if isinstance(model, IsolationForest):
        return -model.score_samples(values)  # Higher means more unusual.
    if isinstance(model, PCA):
        # Compress to principal components, then reconstruct the original features.
        reconstructed = model.inverse_transform(model.transform(values))
        return ((values - reconstructed) ** 2).mean(axis=1)
    # Inference needs predictions only; no gradients or weight updates are needed.
    model.eval()
    with torch.no_grad():
        x = torch.tensor(values, dtype=torch.float32)
        return ((model(x) - x) ** 2).mean(dim=1).numpy()


def train_autoencoder(train, validation, seed):
    # Train a small CPU autoencoder to reconstruct scaled training features.
    # Use MSE loss, Adam, and backpropagation; check separate validation data
    # after every epoch. Return the weights with the lowest accepted validation
    # error, rather than automatically returning the final epoch's weights.

    # Control the random starting weights and shuffled training batches.
    torch.manual_seed(seed)
    size = train.shape[1]
    # Encoder: input -> 16 -> 4; decoder: 4 -> 16 -> input size.
    # The four-value bottleneck compresses patterns; ReLU allows nonlinear learning.
    model = nn.Sequential(nn.Linear(size, 16), nn.ReLU(), nn.Linear(16, 4),
                          nn.ReLU(), nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, size))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.tensor(train, dtype=torch.float32)
    best_loss, best_weights, patience = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        # Shuffle rows each epoch and process batches of at most 256 purchases.
        for indices in torch.randperm(len(x)).split(256):
            batch = x[indices]
            # The target is the input itself: no anomaly labels are used for training.
            loss = ((model(batch) - batch) ** 2).mean()
            # Clear old gradients, calculate new gradients, then update weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation is scored without updating weights; test data are not used here.
        validation_loss = float(anomaly_scores(model, validation).mean())
        # Require improvement greater than 0.00001 and copy the best weights.
        if validation_loss < best_loss - .00001:
            best_loss, best_weights, patience = validation_loss, deepcopy(model.state_dict()), 0
        else:
            patience += 1
        # Stop after 10 consecutive epochs without sufficient validation improvement.
        # An improvement resets the counter; EPOCHS still limits total training.
        if patience >= 10:
            break
    if best_weights is None:
        raise ValueError("Training failed to produce a valid checkpoint.")
    # Restore the saved checkpoint even if the last epoch was worse.
    model.load_state_dict(best_weights)
    print(f"  Autoencoder: {epoch + 1} epochs; best validation MSE = {best_loss:.4f}")
    return model


# 6. EVALUATE ALL MODELS ON EXACTLY THE SAME HELD-OUT BENCHMARK
def run_experiment(orders):
    # Compare Isolation Forest, PCA, and an autoencoder on the same benchmark,
    # first with transaction features and then with added customer history.
    # Print evaluation scores, variation across seeds, and orders for review.
    # Return the first-seed customer autoencoder, its scaler, and its threshold
    # for the optional prediction demo; test scores do not choose this model.

    train, validation, calibration, test = split_data(orders)
    benchmark, labels = make_benchmark(test)
    results, demo = [], None
    # Controlled comparison: use the same splits, benchmark, and training settings.
    # The feature switch changes whether customer history is available.
    for use_history in [False, True]:
        feature_set = "Customer" if use_history else "Transaction"
        # Fit scaling on training data only, then reuse it everywhere else.
        scaler = StandardScaler().fit(make_features(train, use_history))
        x_train, x_val, x_cal, x_test = [scaler.transform(make_features(part, use_history)).astype("float32")
                                      for part in [train, validation, calibration, benchmark]]
        # Same four-dimensional bottleneck as the autoencoder; full PCA is deterministic.
        pca = PCA(n_components=4, svd_solver="full").fit(x_train)
        # Repeat the random models; PCA has no seed variation with this solver.
        for seed in SEEDS:
            print(f"Training {feature_set} models, seed {seed}...")
            baseline = IsolationForest(n_estimators=100, random_state=seed, n_jobs=2).fit(x_train)
            autoencoder = train_autoencoder(x_train, x_val, seed)
            models = [("Isolation Forest", baseline), ("Autoencoder", autoencoder)]
            if seed == SEEDS[0]:
                models.append(("PCA", pca))  # Evaluate deterministic PCA only once.
            for name, model in models:
                # Calibration is separate from early stopping and final testing.
                # The 98th percentile targets roughly the highest-scoring 2% of
                # calibration orders; it does not guarantee a 2% test alert rate.
                threshold = float(np.quantile(anomaly_scores(model, x_cal), .98))
                scores = anomaly_scores(model, x_test)
                predicted = scores > threshold
                # Precision: detected injected anomalies / all flagged benchmark rows.
                # Recall: detected / all injected anomalies; F1 balances both measures.
                # ROC-AUC and average precision summarize ranking across thresholds.
                # OriginalAlertRate is not a verified false-positive rate.
                result = {"Features": feature_set, "Model": name, "Seed": "fixed" if name == "PCA" else seed,
                          "Precision": precision_score(labels, predicted, zero_division=0),
                          "Recall": recall_score(labels, predicted, zero_division=0),
                          "F1": f1_score(labels, predicted, zero_division=0),
                          "ROC_AUC": roc_auc_score(labels, scores),
                          "AvgPrecision": average_precision_score(labels, scores),
                          "OriginalAlertRate": predicted[labels == 0].mean()}
                # Isolation Forest does not reconstruct inputs, so its MSE is N/A.
                for label, title in [(0, "OriginalMSE"), (1, "SyntheticMSE")]:
                    result[title] = scores[labels == label].mean() if name != "Isolation Forest" else np.nan
                # Per-type recall shows which injected behaviors are easier to detect.
                for kind in ["bulk", "high_value", "customer_shift"]:
                    mask = benchmark["kind"].eq(kind).to_numpy()
                    result[kind + "_recall"] = predicted[mask].mean() if mask.any() else np.nan
                results.append(result)
                if use_history and seed == SEEDS[0] and name == "Autoencoder":
                    demo = (model, scaler, threshold)  # Fixed choice, not the best test score.
                    review = test[["customer", "date", "value"]].copy()
                    review["score"] = scores[:len(test)]
    print("\nSYNTHETIC BENCHMARK: original orders are assumed negatives, not verified normal orders.")
    print("Recall is the synthetic anomaly detection rate. MSE applies only to PCA/autoencoders.")
    table = pd.DataFrame(results)
    print(table.round(4).to_string(index=False))
    print("\nMean and standard deviation across training seeds (PCA runs once):")
    # PCA's standard deviation is NaN because it has only one result per feature set.
    # A sample standard deviation requires at least two results.
    print(table.groupby(["Features", "Model"])[["F1", "AvgPrecision"]].agg(["mean", "std"]).round(4))
    print("\nHighest-scoring original orders for human review (not confirmed false positives):")
    print(review.nlargest(5, "score").to_string(index=False))
    return demo


if __name__ == "__main__":
    main()
