import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from preprocessing import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import joblib

#Import scraped data
openf1_data_df = pd.read_csv("openf1_data.csv")

#Number of splits for time series cross validation
n_splits = 8
random_seed = 42

#preprocess data
preprocessed_data = preprocessing(openf1_data_df)

#Class names for the target variable
class_names = {
    -1: "Didn't race",
    1: "Win",
    2: "Podium",
    3: "Top 7",
    4: "Top 10",
    5: "No points"
}

#Number of classes for the target variable
num_classes = len(class_names)

#List of numerical features
numerical_features = ["Session_1_starting_wind_direction",
                      "Session_1_starting_wind_speed",
                      "Session_1_starting_rainfall",
                      "Session_1_starting_track_temperature",
                      "Session_1_starting_air_temperature",
                      "Session_1_starting_humidity",
                      "Session_1_starting_pressure",
                      "Session_1_ending_wind_direction",
                      "Session_1_ending_wind_speed",
                      "Session_1_ending_rainfall",
                      "Session_1_ending_track_temperature",
                      "Session_1_ending_air_temperature",
                      "Session_1_ending_humidity",
                      "Session_1_ending_pressure",
                      "Session_2_starting_wind_direction",
                      "Session_2_starting_wind_speed",
                      "Session_2_starting_rainfall",
                      "Session_2_starting_track_temperature",
                      "Session_2_starting_air_temperature",
                      "Session_2_starting_humidity",
                      "Session_2_starting_pressure",
                      "Session_2_ending_wind_direction",
                      "Session_2_ending_wind_speed",
                      "Session_2_ending_rainfall",
                      "Session_2_ending_track_temperature",
                      "Session_2_ending_air_temperature",
                      "Session_2_ending_humidity",
                      "Session_2_ending_pressure",
                      "Session_3_starting_wind_direction",
                      "Session_3_starting_wind_speed",
                      "Session_3_starting_rainfall",
                      "Session_3_starting_track_temperature",
                      "Session_3_starting_air_temperature",
                      "Session_3_starting_humidity",
                      "Session_3_starting_pressure",
                      "Session_3_ending_wind_direction",
                      "Session_3_ending_wind_speed",
                      "Session_3_ending_rainfall",
                      "Session_3_ending_track_temperature",
                      "Session_3_ending_air_temperature",
                      "Session_3_ending_humidity",
                      "Session_3_ending_pressure",
                      "Session_4_starting_wind_direction",
                      "Session_4_starting_wind_speed",
                      "Session_4_starting_rainfall",
                      "Session_4_starting_track_temperature",
                      "Session_4_starting_air_temperature",
                      "Session_4_starting_humidity",
                      "Session_4_starting_pressure",
                      "Session_4_ending_wind_direction",
                      "Session_4_ending_wind_speed",
                      "Session_4_ending_rainfall",
                      "Session_4_ending_track_temperature",
                      "Session_4_ending_air_temperature",
                      "Session_4_ending_humidity",
                      "Session_4_ending_pressure",
                      "Session_5_starting_wind_direction",
                      "Session_5_starting_wind_speed",
                      "Session_5_starting_rainfall",
                      "Session_5_starting_track_temperature",
                      "Session_5_starting_air_temperature",
                      "Session_5_starting_humidity",
                      "Session_5_starting_pressure"]

#List of categorical features
categorical_features = [
                        "team_id",
                        "driver_number",
                        "circuit_key",
                        "Session_1_session_type_id",
                        "Session_2_session_type_id",
                        "Session_3_session_type_id",
                        "Session_4_session_type_id",
                        "Session_5_session_type_id",
                        "Session_1_position",
                        "Session_2_position",
                        "Session_3_position",
                        "Session_4_position",
                        "Session_1_is_seat_driver",
                        "Session_2_is_seat_driver",
                        "Session_3_is_seat_driver",
                        "Session_4_is_seat_driver",
                        "Session_5_is_seat_driver",
                        "Sprint Wknd",
                        "Race Wknd",
                        "Preseason Wknd"]

#Split data into features and target
X = preprocessed_data[0]
y = preprocessed_data[1]

#Function to get train and test sets for each fold
def get_train_test_set(X, y, idx, label_encoder):
    train_idx, test_idx = idx
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_train, y_test = label_encoder.fit_transform(y_train), label_encoder.transform(y_test)
    return X_train, X_test, y_train, y_test

#Get metrics for final fold
def final_fold_metrics(y_test, y_pred, y_pred_proba, model_type):
    class_metrics = classification_report(y_test, y_pred, target_names=class_names.values()) 
    columns = ["Model Type", "Weighted AUC"]
    columns.extend(class_names.values())
    auc_df = pd.DataFrame(columns=columns)
    # Get AUC scores for each class in final fold
    y_test_binary = np.zeros((len(y_test), num_classes))
    for i in range(num_classes):
        y_test_binary[:, i] = (y_test == i).astype(int)
    
    for i in range(num_classes):
        class_auc = roc_auc_score(y_test_binary[:, i], y_pred_proba[:, i])
        class_name = list(class_names.values())[i]
        auc_df[class_name] = [class_auc]
    auc_df["Model Type"] = [model_type]
    auc_df["Weighted AUC"] = [roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')]
    return auc_df, class_metrics

#Function to train the model
def model_training(X, y, split_indices, model, model_type,use_class_weight = True):
    le = LabelEncoder() #XGboost requires encoded target variable
    print("Training model...")
    for fold, idx in enumerate(split_indices): #Split data into train and test sets for each fold, and train on training set
        print(f"Fold {fold + 1}...")
        X_train, X_test, y_train, y_test = get_train_test_set(X, y, idx, le)
        if use_class_weight:
            class_weights = compute_class_weight( #Compute class weights to account for class imbalance
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_dict = dict(zip(np.unique(y_train), class_weights)) #Create dictionary of class weights
            sample_weights = np.array([class_weights_dict[i] for i in y_train]) #Create array of sample weights
            model.fit(X_train, y_train, classifier__sample_weight=sample_weights) #Train model on training set with sample weights
        else:
            model.fit(X_train, y_train) #Train model on training set without sample weights
        
        y_pred = model.predict(X_test) #Predict on test set
        y_pred_proba = model.predict_proba(X_test) #Predict probabilities on test set
        fold_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted') #Calculate AUC score
        fold_accuracy = accuracy_score(y_test, y_pred) #Calculate accuracy score
        print(f"Fold {fold + 1} AUC: {fold_auc}")
        print(f"Fold {fold + 1} Accuracy: {fold_accuracy}")
    #Get metrics for final fold
    class_aucs, class_metrics = final_fold_metrics(y_test, y_pred, y_pred_proba, model_type)
    print(f"Final Fold Classification Report: \n{class_metrics}")
    print(f"Final Fold AUC: \n{class_aucs}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

XGBoostPipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state= random_seed
    ))
])

tscv = TimeSeriesSplit(n_splits=n_splits)
model_training(X, y, tscv.split(X), XGBoostPipeline, use_class_weight = True, model_type = "XGBoost with Sample Weights")

# Save the entire pipeline
joblib.dump(XGBoostPipeline, 'f1_prediction_model.joblib')

print("Model saved successfully")