import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATASET
ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
portscan = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

df = pd.concat([ddos, portscan], ignore_index=True) #if index not ignored duplicacy and messy
# CLEAN DATA
df.columns = df.columns.str.strip()#removes spaces
df = df.dropna()#nan nat typa stuff from rows
df = df.replace([np.inf, -np.inf], 0)#replaces infinity values with 0
#models cannot handle infinty values

print("Available columns:")
print(df.columns.tolist())#converts column names into a list
#[destination port,flow duration,etc]

# SIMPLE FEATURES THAT EXIST
features = [
    "Destination Port",
    "Packet Length Mean",
    "Fwd Packet Length Max",
    "Total Length of Fwd Packets"
]

#for making a decision the model is going to look at this and use it as its input variables
missing = [col for col in features + ["Label"] if col not in df.columns]
#checks if the column exists otherwise creates a list of columns that are not present in the dataset
# just find all the required columns that are missing from the dataset as well as the features if they are missing 

if missing:
    print("Missing columns:", missing)
    exit() #just stops the execution , just a quick stop not using sys.exit() though should've

X = df[features]
y = df["Label"]

# ENCODE LABELS
#converting the data into a numeric form that a ml model can understand
# sorta like benign -> 0 and ddos->1
le = LabelEncoder() #creates an encoder
y_encoded = le.fit_transform(y)#mapped list
#imp! fit transform is doing two thing its mapping the data to numeric equivalents and converting it likewise
print("Labels found:", list(le.classes_))# LIST OF THE UNIQUE LABELS AND ORDER
#in prediction too gonna use inverse transform to access the labels via the mapped num
# TRAIN TEST SPLIT
#features for training,features for testing,LABELS for training and labels for testing
X_train, X_test, y_train, y_test = train_test_split( #divison into two parts
    X, #question
    y_encoded,#encoded answer
    test_size=0.2, #20 percent data gonna go to testing 80 fo rtraining
    random_state=42,#fixes randomness for debuggin adn consistency
    stratify=y_encoded#same ratio of attacks at both training and testing time to avoid bias
)

# TRAIN MODEL
model = RandomForestClassifier(
    n_estimators=100, #number of trees
    random_state=42, #fixes reandomness same result every time
    n_jobs=-1, #uses all cpu cores for faster training
    class_weight="balanced" #more importance to the attacks otherwise wouldve ignored it beacuse of the actual ratio
)

model.fit(X_train, y_train) #if features then this label

# EVALUATE
y_pred = model.predict(X_test) #new data fed to the modle to determine accuracy

print("Accuracy:", accuracy_score(y_test, y_pred)) #actual vs predicted y_test vs y_pred
print(classification_report(y_test, y_pred, target_names=le.classes_))#how correct the predictions are
# how many actual attacks were detected 
#and balance of both F1 score
# SAVE MODEL FILES
pickle.dump(model, open("model.pkl", "wb")) #pickle module stores in csv wb mode
pickle.dump(le, open("encoder.pkl", "wb"))
pickle.dump(features, open("columns.pkl", "wb"))
#just writing the label mapping , column order


# LIMITATION - feature will only work on lookalike data from mthe dataset
print("New model trained and saved!")
print("Saved files: model.pkl, encoder.pkl, columns.pkl")
