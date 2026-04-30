import pickle
import pandas as pd
import numpy as np
import shap
from collections import defaultdict
from scapy.all import sniff

from flow_builder import update_flow

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

explainer = shap.TreeExplainer(model)

# -----------------------------
# GLOBAL STATS
# -----------------------------
flow_count = 0
label_counts = defaultdict(int)
feature_importance_sum = np.zeros(len(columns))

print("🚀 Real-time Flow-Based XAI IDS started...")
print("Press Ctrl+C to stop and show summary.\n")


# -----------------------------
# SHAP HELPER
# -----------------------------
def get_shap_values_for_prediction(shap_values, prediction):
    arr = np.array(shap_values)

    # New SHAP format: (samples, features, classes)
    if arr.ndim == 3:
        vals = arr[0, :, prediction]

    # Old SHAP format: list[class][sample][feature]
    elif isinstance(shap_values, list):
        vals = np.array(shap_values[prediction][0])

    # Simple format: (samples, features)
    elif arr.ndim == 2:
        vals = arr[0]

    else:
        vals = arr.flatten()

    vals = np.array(vals).flatten()

    # Safety: match feature count
    if len(vals) > len(columns):
        vals = vals[:len(columns)]
    elif len(vals) < len(columns):
        vals = np.pad(vals, (0, len(columns) - len(vals)))

    return vals


# -----------------------------
# PROCESS FLOW
# -----------------------------
def process_packet(packet):
    global flow_count, feature_importance_sum

    try:
        # Flow builder collects packets and returns features only when flow is ready
        features = update_flow(packet)

        if features is None:
            return

        flow_count += 1

        # Keep only model columns
        model_features = {col: features.get(col, 0) for col in columns}
        features_df = pd.DataFrame([model_features], columns=columns)

        # ML prediction
        prediction = model.predict(features_df)[0]
        label = le.inverse_transform([prediction])[0]

        label_counts[label] += 1

        print(f"\nFlow {flow_count} → {label}")

        if label != "BENIGN":
            print("ALERT:", label)

        # XAI explanation using SHAP
        shap_values = explainer.shap_values(features_df)
        vals = get_shap_values_for_prediction(shap_values, prediction)

        feature_importance_sum += np.abs(vals)

        print("Why model predicted this:")

        top_features = sorted(
            zip(columns, vals),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )[:3]

        for feature, value in top_features:
            direction = "supports prediction" if value > 0 else "pushes against prediction"
            print(f"  {feature}: {value:.4f} ({direction})")

        print("-" * 40)

    except Exception as e:
        print("Error while processing packet:", e)


# -----------------------------
# SUMMARY
# -----------------------------
def print_summary():
    print("\n\n ===== IDS SUMMARY =====")
    print(f"\nTotal Flows Analyzed: {flow_count}")

    print("\nTraffic Breakdown:")
    if flow_count == 0:
        print("  No completed flows detected.")
    else:
        for label, count in label_counts.items():
            percent = (count / flow_count) * 100
            print(f"  {label}: {count} ({percent:.2f}%)")

    print("\nTop Influential XAI Features Overall:")

    top_features = sorted(
        zip(columns, feature_importance_sum),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for feature, value in top_features:
        print(f"  {feature}: {value:.4f}")

    print("\n===============================")


# -----------------------------
# START SNIFFING
# -----------------------------
try:
    sniff(prn=process_packet, store=False)

finally:
    print_summary()
