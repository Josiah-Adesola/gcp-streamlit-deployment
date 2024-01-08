#Importing all other Necessary Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
import requests



import pandas as pd

data = "/content/drive/MyDrive/Data/Training.csv"

df = pd.read_csv(data)

df.head()

df.shape
data_test = df.drop(data.index)
data.reset_index(drop=True, inplace=True)
data.head()
data_test.reset_index(drop=True, inplace=True)

data
data['class'].unique()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])
X = data.drop(['class'],axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'eval_metric': 'mlogloss',  # Logarithmic loss for multi-class classification
    'eta': 0.1,  # Learning rate
    'max_depth': 3,  # Maximum depth of the tree
    'subsample': 0.8,  # Subsample ratio of the training instance
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
    'num_class': len(label_encoder.classes_),  # Number of classes
    'seed': 42  # Seed for reproducibility
}

# Train the XGBoost model
num_round = 100
model = xgb.train(params, dtrain, num_round)

y_pred = model.predict(dtest)


accuracy= accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
import pickle

with open('xg_boost_model.pkl', 'wb') as file:
  pickle.dump(model,file)
