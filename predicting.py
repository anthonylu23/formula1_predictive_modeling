import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('f1_position_prediction_model.joblib')

data = pd.read_csv('new_data.csv')
print(data.head())
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

teams = data["team_name"].unique()

teams_dict = {}
for i, team in enumerate(teams):
    teams_dict[team] = i

data["team_id"] = data["team_name"].map(teams_dict)

session_dict = {
    "Practice": 0,
    "Qualifying": 1,
    "Race": 2
}

data["Session_1_session_type_id"] = data["Session_1_session_type"].map(session_dict)
data["Session_2_session_type_id"] = data["Session_2_session_type"].map(session_dict)
data["Session_3_session_type_id"] = data["Session_3_session_type"].map(session_dict)
data["Session_4_session_type_id"] = data["Session_4_session_type"].map(session_dict)
data["Session_5_session_type_id"] = data["Session_5_session_type"].map(session_dict)

seat_drivers = ["VER", "NOR", "BOR", "HAD", "DOO", "GAS", "ANT", "ALO", "LEC", "STR", "TSU", "OCO", "HAM", "SAI", "RUS", "PIA", "BEA", "COL"]

data["Session_1_is_seat_driver"] = data["name_acronym"].isin(seat_drivers).astype(int)
data["Session_2_is_seat_driver"] = data["name_acronym"].isin(seat_drivers).astype(int)
data["Session_3_is_seat_driver"] = data["name_acronym"].isin(seat_drivers).astype(int)
data["Session_4_is_seat_driver"] = data["name_acronym"].isin(seat_drivers).astype(int)
data["Session_5_is_seat_driver"] = data["name_acronym"].isin(seat_drivers).astype(int)

feature_names = ["team_id",
                 "driver_number", 
                 "circuit_key", 
                 "Session_1_session_type_id", 
                 "Session_1_starting_wind_direction",
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
                 "Session_1_position",
                 "Session_2_session_type_id",
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
                 "Session_2_position",
                 "Session_3_session_type_id",
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
                 "Session_3_position",
                 "Session_4_session_type_id",
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
                 "Session_4_position",
                 "Session_5_session_type_id",
                 "Session_5_starting_wind_direction",
                 "Session_5_starting_wind_speed",
                 "Session_5_starting_rainfall",
                 "Session_5_starting_track_temperature",
                 "Session_5_starting_air_temperature",
                 "Session_5_starting_humidity",
                 "Session_5_starting_pressure",
                 "Sprint Wknd",
                 "Race Wknd",
                 "Preseason Wknd",
                 "Session_1_is_seat_driver",
                 "Session_2_is_seat_driver",
                 "Session_3_is_seat_driver",
                 "Session_4_is_seat_driver",
                 "Session_5_is_seat_driver"]
                 
for col in data.columns:
    if col not in feature_names:
        data = data.drop(col, axis=1)

environment_impute = ["Session_1_session_type_id", 
                 "Session_1_starting_wind_direction",
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
                 "Session_2_session_type_id",
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
                 "Session_3_session_type_id",
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
                 "Session_4_session_type_id",
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
                 "Session_5_session_type_id",
                 "Session_5_starting_wind_direction",
                 "Session_5_starting_wind_speed",
                 "Session_5_starting_rainfall",
                 "Session_5_starting_track_temperature",
                 "Session_5_starting_air_temperature",
                 "Session_5_starting_humidity",
                 "Session_5_starting_pressure",
                 "Session_1_is_seat_driver",
                 "Session_2_is_seat_driver",
                 "Session_3_is_seat_driver",
                 "Session_4_is_seat_driver",
                 "Session_5_is_seat_driver"]

# Create a boolean mask for rows where 'circuit_key' is the same as the previous row
is_same_circuit = data['circuit_key'] == data['circuit_key'].shift(1)

for feature in environment_impute:
    # Check if the feature exists in the DataFrame
    if feature in data.columns:
        # Create a mask for nulls in the feature that should be forward-filled
        ffill_mask = data[feature].isnull() & is_same_circuit
        
        # Get the forward-filled values for the masked locations and assign them
        data.loc[ffill_mask, feature] = data[feature].ffill()[ffill_mask]
        
        # For the remaining nulls, impute with the value from the following entry (backward fill)
        data[feature].fillna(method='bfill', inplace=True)


position_features = ["Session_1_position", "Session_2_position", "Session_3_position", "Session_4_position"]

for feature in position_features:
    # Check if the feature exists in the DataFrame
    if feature in data.columns:
        # Fill missing values with -1, indicating the driver did not participate or finish
        data[feature] = data[feature].fillna(-1)

X = data[feature_names]

# Ensure X is numeric and handle any remaining data type issues
# Convert to numeric, coercing errors to NaN, then fill NaN with 0
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

X[["team_id", "driver_number", "circuit_key", "Sprint Wknd", "Race Wknd", "Preseason Wknd"]] = X[["team_id", "driver_number", "circuit_key", "Sprint Wknd", "Race Wknd", "Preseason Wknd"]].astype(object)
# types = X.dtypes
# for col, type in types.items():
#     print(col, type)

print(X.head())
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
results = pd.DataFrame(columns=["name_acronym", "driver_number", "Qual Pos", "prediction"])

results["name_acronym"] = X["driver_number"].map(f1_drivers_2025)
results["driver_number"] = X["driver_number"]
results["Qual Pos"] = X["Session_4_position"]
results["prediction"] = predictions

position_unbin = {
    1: "Win - 25pts",
    2: "Podium - 15pts",
    3: "Top 7 - 5pts",
    4: "Top 10 - 1pts",
    5: "No Pts",
}

results["prediction"] = results["prediction"].map(position_unbin)

print(results.sort_values(by="Qual Pos", ascending=True))