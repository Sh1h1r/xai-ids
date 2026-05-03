# XAI-Based Intrusion Detection System (IDS)

## Overview

This project implements a real-time Intrusion Detection System (IDS) using Machine Learning and Explainable AI (XAI).

It captures live network traffic, converts packets into flows, predicts whether the traffic is benign or a DDoS or portscan attack, and explains each prediction using SHAP (SHapley Additive Explanations).

---

## Key Features

- Real-time packet capture using Scapy  
- Flow-based feature extraction  
- Machine learning model (Random Forest)  
- Explainable AI using SHAP  
- Graphical user interface using Tkinter  
- Dataset-based demo mode for reliable DDoS detection  

---

## Tech Stack

- Python  
- Scapy  
- Pandas / NumPy  
- Scikit-learn  
- SHAP  
- Tkinter  

---

## Project Structure

XAI-IDS/
├── app1.py
├── real_time.py
├── flow_builder.py
├── train.py
├── demo_detection.py
├── model.pkl
├── encoder.pkl
├── columns.pkl

---

## How to Run

### 1. Train the Model (optional)

python3 train.py

### 2. Run GUI (recommended)

sudo python3 app1.py

### 3. Run Real-Time IDS (terminal mode)

sudo python3 real_time.py

### 4. Run Dataset Demo (for DDoS detection and XAI)

python3 demo_detection.py

---

## System Pipeline

Network Traffic → Packet Capture → Flow Builder → ML Model → Prediction → SHAP Explanation

---

## Prediction Classes

- BENIGN  
- DDoS
- PortScan

---

## Explainability (XAI)

Each prediction is accompanied by feature-level explanations:

- Positive SHAP value: supports the prediction  
- Negative SHAP value: pushes against the prediction  

Example:

Flow → BENIGN

Why:
Packet Length Mean → supports prediction  
Destination Port → pushes against prediction  

---

## Important Notes

- The model is trained on the CICIDS2017 dataset (flow-based data)  
- Real-time traffic is packet-based and may not match dataset patterns  
- Real-time mode demonstrates live traffic capture  
- Demo mode demonstrates accurate DDoS detection and explainability  

---

## Limitations

- Real-time detection accuracy depends on similarity to training data  
- Flow features are approximated using a simplified flow builder  
- The system currently detects only BENIGN and DDoS traffic  

---

## Future Improvements

- Integrate CICFlowMeter for accurate flow feature extraction  
- Extend detection to additional attack types  
- Improve real-time detection accuracy  
- Add visualization dashboards  

---

## Author

Shihir Yadav
