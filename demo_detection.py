import pandas as pd
import numpy as np
import pickle
import shap
import time

# Load everything
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

explainer = shap.TreeExplainer(model)

# Load dataset
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.columns = df.columns.str.strip()
df = df.replace([np.inf, -np.inf], 0)
df = df.dropna()

X = df[columns]
y = df["Label"]

print("DEMO MODE: Showing real detections with explanations\n")

# Loop through some samples
for i in range(20):
    sample = X.iloc[i:i+1]

    prediction = model.predict(sample)[0]
    label = le.inverse_transform([prediction])[0]

    print(f"Sample {i} → {label}")

    # SHAP explanation
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        vals = shap_values[prediction][0]
    else:
        vals = shap_values[0]

    vals = np.array(vals).flatten()

    top_features = sorted(
        zip(columns, vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    print("   Why:")
    for f, v in top_features:
        print(f"   {f}: {v:.4f}")

    print("-" * 40)
    time.sleep(1)
