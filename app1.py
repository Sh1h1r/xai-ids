import tkinter as tk
from tkinter import scrolledtext
import threading
import pickle
import pandas as pd
import numpy as np
import shap
from collections import defaultdict
from scapy.all import sniff

from flow_builder import update_flow

model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

explainer = shap.TreeExplainer(model)

running = False
flow_count = 0
label_counts = defaultdict(int)
feature_importance_sum = np.zeros(len(columns))

BG_DARK = "#0a0d12"
BG_PANEL = "#0f1319"
BG_CARD = "#141920"
BORDER = "#1e2a38"
ACCENT_CYAN = "#00d4ff"
ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff3d5a"
ACCENT_AMBER = "#ffb300"
TEXT_PRIMARY = "#e8edf2"
TEXT_MUTED = "#4a5d72"
TEXT_DIM = "#2a3a4a"

root = tk.Tk()
root.title("XAI Intrusion Detection System")
root.geometry("960x720")
root.configure(bg=BG_DARK)
root.resizable(True, True)

FONT_MONO = ("Courier New", 10)
FONT_HEAD = ("Courier New", 18, "bold")
FONT_LABEL = ("Courier New", 11, "bold")
FONT_BTN = ("Courier New", 10, "bold")
FONT_STATUS = ("Courier New", 9)

header_frame = tk.Frame(root, bg=BG_PANEL, height=72)
header_frame.pack(fill="x")
header_frame.pack_propagate(False)

accent_bar = tk.Frame(header_frame, bg=ACCENT_CYAN, width=4)
accent_bar.pack(side="left", fill="y")

header_inner = tk.Frame(header_frame, bg=BG_PANEL)
header_inner.pack(side="left", fill="both", expand=True, padx=18, pady=10)

title_row = tk.Frame(header_inner, bg=BG_PANEL)
title_row.pack(anchor="w")

tk.Label(title_row, text="[", font=FONT_HEAD, fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")
tk.Label(title_row, text=" XAI-IDS ", font=FONT_HEAD, fg=TEXT_PRIMARY, bg=BG_PANEL).pack(side="left")
tk.Label(title_row, text="]", font=FONT_HEAD, fg=ACCENT_CYAN, bg=BG_PANEL).pack(side="left")

tk.Label(
    header_inner,
    text="Explainable AI  •  Real-Time Network Intrusion Detection",
    font=FONT_STATUS,
    fg=TEXT_MUTED,
    bg=BG_PANEL
).pack(anchor="w")

status_right = tk.Frame(header_frame, bg=BG_PANEL)
status_right.pack(side="right", padx=18, pady=12)

dot_canvas = tk.Canvas(status_right, width=12, height=12, bg=BG_PANEL, highlightthickness=0)
dot_canvas.pack(side="left", padx=(0, 6))
status_dot = dot_canvas.create_oval(1, 1, 11, 11, fill=ACCENT_RED, outline="")

status_label = tk.Label(status_right, text="OFFLINE", font=FONT_LABEL, fg=ACCENT_RED, bg=BG_PANEL)
status_label.pack(side="left")

tk.Frame(root, bg=ACCENT_CYAN, height=1).pack(fill="x")

stats_frame = tk.Frame(root, bg=BG_DARK)
stats_frame.pack(fill="x", padx=14, pady=(10, 4))


def make_stat_card(parent, label_text, value_text, accent):
    card = tk.Frame(parent, bg=BG_CARD)
    card.pack(side="left", expand=True, fill="x", padx=4)

    top_line = tk.Frame(card, bg=accent, height=2)
    top_line.place(x=0, y=0, relwidth=1)

    inner = tk.Frame(card, bg=BG_CARD)
    inner.pack(padx=14, pady=10, anchor="w")

    tk.Label(inner, text=label_text, font=FONT_STATUS, fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

    val = tk.Label(inner, text=value_text, font=FONT_LABEL, fg=accent, bg=BG_CARD)
    val.pack(anchor="w")

    return val


flow_val = make_stat_card(stats_frame, "FLOWS ANALYZED", "0", ACCENT_CYAN)
alert_val = make_stat_card(stats_frame, "THREATS DETECTED", "0", ACCENT_RED)
benign_val = make_stat_card(stats_frame, "BENIGN FLOWS", "0", ACCENT_GREEN)
rate_val = make_stat_card(stats_frame, "ENGINE STATUS", "IDLE", ACCENT_AMBER)

alert_count = 0
benign_count = 0

content_frame = tk.Frame(root, bg=BG_DARK)
content_frame.pack(fill="both", expand=True, padx=14, pady=(4, 0))

log_header = tk.Frame(content_frame, bg=BG_CARD)
log_header.pack(fill="x")

tk.Label(
    log_header,
    text="▸ LIVE TRAFFIC LOG",
    font=FONT_LABEL,
    fg=ACCENT_CYAN,
    bg=BG_CARD
).pack(side="left", padx=14, pady=6)

tk.Label(
    log_header,
    text="[ scroll to review ]",
    font=FONT_STATUS,
    fg=TEXT_DIM,
    bg=BG_CARD
).pack(side="right", padx=14)

output_frame = tk.Frame(content_frame, bg=BORDER, bd=1)
output_frame.pack(fill="both", expand=True, pady=(1, 0))

output = scrolledtext.ScrolledText(
    output_frame,
    bg=BG_PANEL,
    fg=ACCENT_GREEN,
    insertbackground=ACCENT_CYAN,
    selectbackground="#003344",
    selectforeground=TEXT_PRIMARY,
    font=FONT_MONO,
    relief="flat",
    bd=0,
    padx=14,
    pady=10,
    wrap="word"
)
output.pack(fill="both", expand=True)

output.tag_config("alert", foreground=ACCENT_RED, font=("Courier New", 10, "bold"))
output.tag_config("benign", foreground=ACCENT_GREEN)
output.tag_config("header", foreground=ACCENT_CYAN, font=("Courier New", 10, "bold"))
output.tag_config("muted", foreground=TEXT_MUTED)
output.tag_config("warning", foreground=ACCENT_AMBER)
output.tag_config("dim", foreground=TEXT_DIM)
output.tag_config("normal", foreground=ACCENT_GREEN)

tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(4, 0))

button_frame = tk.Frame(root, bg=BG_DARK)
button_frame.pack(fill="x", padx=14, pady=10)


def log(text, tag="normal"):
    output.insert(tk.END, text + "\n", tag)
    output.see(tk.END)


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


def process_packet(packet):
    global flow_count, feature_importance_sum, alert_count, benign_count

    if not running:
        return

    try:
        features = update_flow(packet)

        if features is None:
            return

        flow_count += 1

        model_features = {col: features.get(col, 0) for col in columns}
        features_df = pd.DataFrame([model_features], columns=columns)

        prediction = model.predict(features_df)[0]
        label = le.inverse_transform([prediction])[0]

        label_counts[label] += 1

        is_attack = label != "BENIGN"

        if is_attack:
            alert_count += 1
        else:
            benign_count += 1

        flow_val.config(text=str(flow_count))
        alert_val.config(text=str(alert_count))
        benign_val.config(text=str(benign_count))

        shap_values = explainer.shap_values(features_df)
        vals = get_shap_values_for_prediction(shap_values, prediction)

        feature_importance_sum += np.abs(vals)

        tag = "alert" if is_attack else "benign"

        log(f"\n{'─' * 48}", "dim")
        log(f"  FLOW #{flow_count:04d}  →  {label}", tag)

        if is_attack:
            log(f"  THREAT DETECTED: {label}", "alert")

        log("  XAI Explanation:", "header")

        top_features = sorted(
            zip(columns, vals),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )[:3]

        for feature, value in top_features:
            direction = "supports prediction" if value > 0 else "pushes against prediction"
            feat_tag = "warning" if value > 0 else "muted"
            log(f"    {feature:<30} {value:+.4f}  {direction}", feat_tag)

    except Exception as e:
        log(f"  [ERROR] {e}", "alert")


def sniff_loop():
    while running:
        sniff(prn=process_packet, store=False, timeout=1)


def start_ids():
    global running

    if running:
        return

    running = True

    status_label.config(text="ACTIVE", fg=ACCENT_GREEN)
    dot_canvas.itemconfig(status_dot, fill=ACCENT_GREEN)
    rate_val.config(text="RUNNING")

    log("XAI-IDS ENGINE STARTED", "header")
    log("Listening for network traffic...\n", "muted")

    thread = threading.Thread(target=sniff_loop, daemon=True)
    thread.start()


def stop_ids():
    global running

    running = False

    status_label.config(text="OFFLINE", fg=ACCENT_RED)
    dot_canvas.itemconfig(status_dot, fill=ACCENT_RED)
    rate_val.config(text="IDLE")

    log(f"\n{'═' * 48}", "header")
    log("SESSION SUMMARY", "header")
    log(f"{'═' * 48}", "header")

    log(f"Total Flows Analyzed : {flow_count}", "normal")
    log(f"Threats Detected     : {alert_count}", "alert" if alert_count else "muted")
    log(f"Benign Flows         : {benign_count}", "benign")

    log("\nTraffic Breakdown:", "header")

    if flow_count == 0:
        log("No completed flows detected.", "muted")
    else:
        for label, count in label_counts.items():
            percent = (count / flow_count) * 100
            tag = "alert" if label != "BENIGN" else "benign"
            log(f"{label:<30} {count:>5}  ({percent:.1f}%)", tag)

    log("\nTop XAI Feature Influences:", "header")

    top_features = sorted(
        zip(columns, feature_importance_sum),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for i, (feature, value) in enumerate(top_features, 1):
        log(f"{i}. {feature:<30} {value:.4f}", "warning")

    log(f"{'═' * 48}\n", "header")


def clear_output():
    output.delete("1.0", tk.END)


def run_dataset_demo():
    try:
        log("\nDATASET XAI DEMO", "header")
        log("Loading DDoS and PortScan datasets...\n", "muted")

        ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        portscan = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

        df = pd.concat([ddos, portscan], ignore_index=True)

        df.columns = df.columns.str.strip()
        df = df.replace([np.inf, -np.inf], 0)
        df = df.dropna()

        log(f"Model Labels: {list(le.classes_)}", "muted")

        samples_per_class = 5
        all_samples = []

        for label_name in le.classes_:
            class_df = df[df["Label"] == label_name]

            if len(class_df) == 0:
                log(f"No samples found for label: {label_name}", "alert")
                continue

            n = min(samples_per_class, len(class_df))
            sampled = class_df.sample(n=n, random_state=42)
            all_samples.append(sampled)

        if not all_samples:
            log("No samples available for demo.", "alert")
            return

        samples = pd.concat(all_samples, ignore_index=True)

        for i in range(len(samples)):
            sample = samples.iloc[i:i + 1]
            actual_label = sample["Label"].iloc[0]

            X_sample = sample[columns]

            prediction = model.predict(X_sample)[0]
            predicted_label = le.inverse_transform([prediction])[0]

            shap_values = explainer.shap_values(X_sample)
            vals = get_shap_values_for_prediction(shap_values, prediction)

            is_attack = actual_label != "BENIGN"
            tag = "alert" if is_attack else "benign"
            match = "MATCH" if actual_label == predicted_label else "MISMATCH"

            log(f"\n{'─' * 46}", "dim")
            log(f"Sample {i + 1:02d} [{match}]", "header")
            log(f"Actual    : {actual_label}", tag)
            log(f"Predicted : {predicted_label}", tag)
            log("XAI Features:", "header")

            top_features = sorted(
                zip(columns, vals),
                key=lambda x: abs(float(x[1])),
                reverse=True
            )[:3]

            for feature, value in top_features:
                direction = "supports prediction" if value > 0 else "pushes against prediction"
                log(f"  {feature:<30} {value:+.4f}  {direction}", "warning")

        log(f"\n{'─' * 46}", "dim")
        log("Demo complete.\n", "header")

    except Exception as e:
        log(f"[DEMO ERROR] {e}", "alert")


def make_button(parent, text, command, accent, col):
    outer = tk.Frame(parent, bg=accent, padx=1, pady=1)
    outer.grid(row=0, column=col, padx=6)

    btn = tk.Button(
        outer,
        text=text,
        command=command,
        font=FONT_BTN,
        bg=BG_CARD,
        fg=accent,
        activebackground=accent,
        activeforeground=BG_DARK,
        relief="flat",
        bd=0,
        padx=16,
        pady=8,
        cursor="hand2",
        width=18
    )
    btn.pack()

    def on_enter(e):
        btn.config(bg=accent, fg=BG_DARK)

    def on_leave(e):
        btn.config(bg=BG_CARD, fg=accent)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn


make_button(button_frame, "START IDS", start_ids, ACCENT_GREEN, 0)
make_button(button_frame, "STOP + SUMMARY", stop_ids, ACCENT_RED, 1)
make_button(button_frame, "DATASET DEMO", run_dataset_demo, ACCENT_CYAN, 2)
make_button(button_frame, "CLEAR LOG", clear_output, ACCENT_AMBER, 3)

footer = tk.Frame(root, bg=BG_PANEL, height=24)
footer.pack(fill="x")
footer.pack_propagate(False)

tk.Label(
    footer,
    text="XAI-IDS v1.0  •  SHAP-powered explanations  •  Real-time packet capture",
    font=FONT_STATUS,
    fg=TEXT_DIM,
    bg=BG_PANEL
).pack(side="left", padx=16)

log("XAI-IDS initialized. Press START to begin monitoring.", "header")
log("Powered by SHAP explainability + ML threat classification.\n", "muted")

root.mainloop()
