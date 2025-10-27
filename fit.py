from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import joblib

# Importing the dataset
def data_reading(path = "datasets/extracted_audio_features.csv"):
  df_raw = pd.read_csv(path)

  # Preprocessing
  x_df = df_raw.iloc[:,:-1]
  y_nacoded = df_raw.iloc[:,-1]

  le = LabelEncoder()
  y = le.fit_transform(y_nacoded)

  feature_names = x_df.columns.tolist()
  x = x_df.to_numpy()
  return x,y,feature_names

def model():
  # Train-test split
  x,y,feature_names = data_reading()
  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=1/3,shuffle=True,random_state=14,stratify=y
  )

  # Base models
  rf_model = RandomForestClassifier(class_weight="balanced",random_state=92)
  svc_model = SVC(class_weight="balanced",random_state=94)
  lr_model = LogisticRegression(random_state=97,class_weight="balanced")

  # Pipelines
  preprocessor = Pipeline([
    ("imputation",SimpleImputer(strategy="median")),
    ("scale",StandardScaler())
  ])
  pipe = Pipeline([
    ("preprocessor",preprocessor),
    ("pca",PCA(random_state=192)),
    ("model",rf_model)
  ])

  # Parameter Configurations for GridSsearchCV
  param_dict = [
  { #Random Forest, PCA On
    "model": [rf_model], "model__n_estimators": np.arange(500,1000,100),
    "model__max_depth":np.arange(10,25,2), "pca__n_components":np.arange(15,32,2)
  },
  { #SVC_rbf, PCA On
    "model" : [svc_model], "model__kernel":["rbf"],
    "model__C":[0.001,0.01,0.1,1,10,100,10**3],
    "model__gamma" : [0.001,0.01,0.1,1,10,100,10**3],
    "pca__n_components":np.arange(15,32,2)
  },
  { #LogisticRegression, PCA On, lbfgs solver, l2 penalty
    "model" : [lr_model], "model__C":[0.001,0.01,0.1,1,10,100,10**3],
    "model__penalty": ["l2"],"model__solver":["lbfgs"],
    "pca__n_components":np.arange(15,32,2)
  },
  { #LogisticRegression, PCA On, newton-cg solver, l2 penalty
    "model" : [lr_model], "model__C":[0.001,0.01,0.1,1,10,100,10**3],
    "model__penalty": ["l2"],"model__solver":["newton-cg"],
    "pca__n_components":np.arange(15,32,2)
  },
  {  #Random Forest, PCA Off
    "model": [rf_model], "model__n_estimators": np.arange(500,1000,100),
    "model__max_depth":np.arange(10,25,2), "pca":[None]
  }
  ]

  gs = RandomizedSearchCV(
    pipe,param_distributions=param_dict,n_iter=15,refit=True,n_jobs=-1,cv=7,
    random_state=82
  )
  print("Starting Model Training......")
  print("It will take some time to train, so please wait")
  t1 = time.time()
  gs.fit(x_train,y_train)
  t2 = time.time()
  minutes,seconds = divmod(t2-t1,60)
  print("✅ Model Trained Successfully")
  print(f"Time elapsed: {int(minutes)} Minute {seconds:.2f} Seconds")
  pipe_best = gs.best_estimator_
  return pipe_best

# Model Evaluation
def evaluation(pipe):
  x,y,feature_names = data_reading()
  x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=1/3,shuffle=True,random_state=14,stratify=y
  )
  y_true = y_test
  y_pred = pipe.predict(x_test)
  print(classification_report(y_true=y_true,y_pred=y_pred))

# Dumping the model in pickle files
def dumping(pipe, feature_names):
  joblib.dump(pipe,"models/pipe.pkl")
  joblib.dump(feature_names,"models/feature_names.pkl")
  print("✅ .pkl files are dumped successfully")

def main():
  pipe = model()
  evaluation(pipe)
  # Reread data to get feature names for dumping without re-training
  x,y,feature_names = data_reading()
  dumping(pipe, feature_names)

if __name__ == "__main__":
  main()