import pickle
import pandas as pd
import numpy as np
import shap
from collections import defaultdict
from scapy.all import sniff

from flow_builder import update_flow

#
# LOAD MODEL FILES
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb")) #encoded stuff to normal
columns = pickle.load(open("columns.pkl", "rb")) #list of features needed

explainer = shap.TreeExplainer(model)

# GLOBAL STATS

flow_count = 0 #to count flow
label_counts = defaultdict(int) #dictionary for how many times an attack occured
feature_importance_sum = np.zeros(len(columns)) #to understand how important a feature is by keeping a track of all

print(" Real-time Flow-Based XAI IDS started...")
print("Press Ctrl+C to stop and show summary.\n")


# SHAP HELPER
#This function ensures that no matter how SHAP gives us the data, we turn it into a simple, flat list that matches our columns.
def get_shap_values_for_prediction(shap_values, prediction):

#so we first make an array of the explantion made by the shap and then grab only the first row cuz its the ongoing flow and the make a prediction based on that and then classify it
    arr = np.array(shap_values) #flattens the data into array to check the layers of data

    # New SHAP format: (samples, features, classes)
    if arr.ndim == 3:
        vals = arr[0, :, prediction] #to remove 3dness of the shap and just get the 0th samples and all features with prediciton
#cuz shap is made to handle large rows and we only have one flow goig at a time so just the 0th row
    # Old SHAP format: list[class][sample][feature]
    elif isinstance(shap_values, list): #checking its an array 
        vals = np.array(shap_values[prediction][0])
        #just grabbing the first row cuz its the ongoing flow and classify the attack based on prediction

    # Simple format: (samples, features) #final check becasue the shape of DATA HAS TO MATCH MODEL'S COLUMNS
    elif arr.ndim == 2:
        vals = arr[0] #when only two dimnesin r there its a table in 3 its a cube so we are just generalising the data

    else:
        vals = arr.flatten() #even if the format is still weird we just flatten everything to an array and just grab the first flow
    vals = np.array(vals).flatten() #jsut checkcing if its def a 1d array

    # Safety: match feature count
    if len(vals) > len(columns):    
        vals = vals[:len(columns)] #slcing the list because shap soemtimes includes base value messig up the data had me run into a bug
    elif len(vals) < len(columns):
        vals = np.pad(vals, (0, len(columns) - len(vals))) #if by chance the array is short we just add 0s at teh end to emake it of the requi leength

    return vals

# PROCESS FLOW
def process_packet(packet):
    global flow_count, feature_importance_sum #gotta update the import feature count

    try:
        # Flow builder collects packets and returns features only when flow is ready
        features = update_flow(packet) #
    #goes to flow_builder to get categorized
        if features is None:
            return
#waiting till the 3 second window is done
        flow_count += 1
#increasing flow count each time its finished 
        # Keep only model columns
        model_features = {col: features.get(col, 0) for col in columns} #cleaning data if value exists cool i fnot 0 is put in this dict
        features_df = pd.DataFrame([model_features], columns=columns) #model expects a table

        # ML prediction
        prediction = model.predict(features_df)[0] #grab first row only make prediction on the first row
        label = le.inverse_transform([prediction])[0] #encoded stuff back into labels like in train.py g

        label_counts[label] += 1 #global tally updated

        print(f"\nFlow {flow_count} → {label}")

        if label != "BENIGN":
            print("ALERT:", label)

        # XAI explanation using SHAP
        shap_values = explainer.shap_values(features_df) #The SHAP explainer takes the flow data and calculates a score for every single feature (Port, Packet Length, etc.). These scores represent how much each feature contributed to the model's decision.
        vals = get_shap_values_for_prediction(shap_values, prediction)
#shap is a weirdo returns stuff messy we just want a list dawg
        feature_importance_sum += np.abs(vals)
#"Importance of the Flow" refers to how much weight the AI model gives to a specific set of network behaviors when deciding if those behaviors are malicious or safe.
#without this we would know the attack but we wont know how it is being attacked global importance keeps a track of it 
        print("Why model predicted this:")

        top_features = sorted(
            zip(columns, vals), #pairs features with scores
            key=lambda x: abs(float(x[1])), #sort on basis of absoulte we just see the move from the inti 0 
            reverse=True #biggest at top 
        )[:3] #slices the list for top 3 

        for feature, value in top_features: #looks at top 3 and makes a decision 
            direction = "supports prediction" if value > 0 else "pushes against prediction"
            print(f"  {feature}: {value:.4f} ({direction})")

        print("-" * 40)

    except Exception as e:
        print("Error while processing packet:", e)

# SUMMARY
def print_summary():
    print("\n\n ===== IDS SUMMARY =====")
    print(f"\nTotal Flows Analyzed: {flow_count}")

    print("\nTraffic Breakdown:")
    if flow_count == 0: #checks if the program didnt stop at 3 second window
        print("  No completed flows detected.")
    else:
        for label, count in label_counts.items():
            percent = (count / flow_count) * 100
            print(f"  {label}: {count} ({percent:.2f}%)")

    print("\nTop Influential XAI Features Overall:")

    top_features = sorted(
        zip(columns, feature_importance_sum), #shows in pairs 
        key=lambda x: x[1], #sort on the basis of feature importance
        reverse=True #smallest to largest
    )[:5] #top 5 

    for feature, value in top_features:
        print(f"  {feature}: {value:.4f}")

    print("\n===============================")

# START SNIFFING
try:
    sniff(prn=process_packet, store=False)

finally:
    print_summary()
