import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATASET
df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# CLEAN DATA
df.columns = df.columns.str.strip()
df = df.dropna()
df = df.replace([np.inf, -np.inf], 0)

print("Available columns:")
print(df.columns.tolist())

# SIMPLE FEATURES THAT EXIST
features = [
    "Destination Port",
    "Packet Length Mean",
    "Fwd Packet Length Max",
    "Total Length of Fwd Packets"
]

missing = [col for col in features + ["Label"] if col not in df.columns]

if missing:
    print("Missing columns:", missing)
    exit()

X = df[features]
y = df["Label"]

# ENCODE LABELS
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Labels found:", list(le.classes_))

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)
# EVALUATE
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
# SAVE MODEL FILES
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))
pickle.dump(features, open("columns.pkl", "wb"))

print("New model trained and saved!")
print("Saved files: model.pkl, encoder.pkl, columns.pkl")
