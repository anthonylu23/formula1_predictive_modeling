import joblib
import pandas as pd
import numpy as np
from preprocessing import preprocessing

# Load the trained model
model = joblib.load('f1_prediction_model.joblib')

#Load unseen data that we want to make predictions on
data = pd.read_csv('new_data.csv')
print(data.head())

#Set conditions for the race
session_5_conditions = {
    "Session_5_session_type_id": 2,
    "Session_5_starting_wind_direction": 22.5,
    "Session_5_starting_wind_speed": 8,
    "Session_5_starting_rainfall": 0,
    "Session_5_starting_track_temperature": 0,
    "Session_5_starting_air_temperature": 31,
    "Session_5_starting_humidity": 32,
    "Session_5_starting_pressure": 1024,
}

for col, value in session_5_conditions.items():
    data[col] = float(value)

data["Session_5_session_type"] = "Race"

#Preprocess data
preprocessed_data = preprocessing(data)

X = preprocessed_data[0]

#Make predictions
predictions = model.predict(X)
f1_drivers_2025 = {
    1: "VER",  # Max Verstappen
    4: "NOR",  # Lando Norris
    5: "BOR",  # Gabriel Bortoleto
    6: "HAD",  # Isack Hadjar
    10: "GAS", # Pierre Gasly
    12: "ANT", # Andrea Kimi Antonelli
    14: "ALO", # Fernando Alonso
    16: "LEC", # Charles Leclerc
    18: "STR", # Lance Stroll
    22: "TSU", # Yuki Tsunoda
    23: "ALB", # Alex Albon
    27: "HUL", # Nico Hülkenberg
    30: "LAW", # Liam Lawson
    31: "OCO", # Esteban Ocon
    43: "COL", # Franco Colapinto (from 2024, expected to continue)
    44: "HAM", # Lewis Hamilton
    55: "SAI", # Carlos Sainz
    63: "RUS", # George Russell
    81: "PIA", # Oscar Piastri
    87: "BEA"  # Oliver Bearman
}

#Create dataframe with predictions
results = pd.DataFrame(columns=["name_acronym", "driver_number", "Qual Pos", "prediction"])

results["name_acronym"] = X["driver_number"].map(f1_drivers_2025)
results["driver_number"] = X["driver_number"]
results["Qual Pos"] = X["Session_4_position"]
results["prediction"] = predictions

position_unbin = {
    1: "Win",
    2: "Podium",
    3: "Top 7",
    4: "Top 10",
    5: "No points",
}

results["prediction"] = results["prediction"].map(position_unbin)

print(results.sort_values(by="Qual Pos", ascending=True))