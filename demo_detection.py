import pandas as pd
import numpy as np
import pickle
import shap
import time

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

explainer = shap.TreeExplainer(model)

# -----------------------------
# LOAD DATASETS
# -----------------------------
ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
portscan = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

df = pd.concat([ddos, portscan], ignore_index=True)

df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], 0)
df = df.dropna()

print("DEMO MODE: Multi-class detection with XAI explanations\n")
print("Labels in model:", list(le.classes_))
print("-" * 50)


# -----------------------------
# SHAP HELPER
# -----------------------------
def get_shap_values_for_prediction(shap_values, prediction):
    arr = np.array(shap_values)

    if arr.ndim == 3:
        vals = arr[0, :, prediction]
    elif isinstance(shap_values, list):
        vals = np.array(shap_values[prediction][0])
    elif arr.ndim == 2:
        vals = arr[0]
    else:
        vals = arr.flatten()

    vals = np.array(vals).flatten()

    if len(vals) > len(columns):
        vals = vals[:len(columns)]
    elif len(vals) < len(columns):
        vals = np.pad(vals, (0, len(columns) - len(vals)))

    return vals

#same bruh
# -----------------------------
# SAMPLE SETTINGS
# -----------------------------
samples_per_class = 5

all_samples = []

for label_name in le.classes_:
    class_df = df[df["Label"] == label_name]

    if len(class_df) == 0:
        print(f"No samples found for label: {label_name}")
        continue

    n = min(samples_per_class, len(class_df))
    sampled = class_df.sample(n=n, random_state=42)

    all_samples.append(sampled)

demo_df = pd.concat(all_samples, ignore_index=True)


# -----------------------------
# RUN DEMO
# -----------------------------
for i in range(len(demo_df)):
    sample = demo_df.iloc[i:i + 1]
    actual_label = sample["Label"].iloc[0]

    X_sample = sample[columns]

    prediction = model.predict(X_sample)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    print(f"Sample {i + 1}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {predicted_label}")

    shap_values = explainer.shap_values(X_sample)
    vals = get_shap_values_for_prediction(shap_values, prediction)

    top_features = sorted(
        zip(columns, vals),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[:3]

    print("Why model predicted this:")
    for feature, value in top_features:
        direction = "supports prediction" if value > 0 else "pushes against prediction"
        print(f"  {feature}: {value:.4f} ({direction})")

    print("-" * 50)
    time.sleep(1)
