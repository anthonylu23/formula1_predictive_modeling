import numpy as np
import pandas as pd

# Drivers that were signed during this season
# DOO no longer contracted with Alpine, replaced by COL
# TSU and LAW swapped (RBR <-> VCARB)
signed_drivers = ["VER", "NOR", "BOR", "HAD", "DOO", "GAS", "ANT", "ALO", "LEC", "STR", "TSU", "OCO", "HAM", "SAI", "RUS", "PIA", "BEA", "COL"]

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

#Function to encode constructor names to integers
def constructor_encoding(data_df):
    teams = data_df["team_name"].unique()

    teams_dict = {}
    for i, team in enumerate(teams):
        teams_dict[team] = i

    data_df["team_id"] = data_df["team_name"].map(teams_dict)
    return data_df

#Function to encode session types to integers
def session_type_encoding(data_df):
    session_dict = {
        "Practice": 0,
        "Qualifying": 1,
        "Race": 2
    }

    data_df["Session_1_session_type_id"] = data_df["Session_1_session_type"].map(session_dict)
    data_df["Session_2_session_type_id"] = data_df["Session_2_session_type"].map(session_dict)
    data_df["Session_3_session_type_id"] = data_df["Session_3_session_type"].map(session_dict)
    data_df["Session_4_session_type_id"] = data_df["Session_4_session_type"].map(session_dict)
    data_df["Session_5_session_type_id"] = data_df["Session_5_session_type"].map(session_dict)    

    return data_df

#Function to encode race results to integers
def race_result_encoding(data_df):
    def bin_finishing_position(position):
        if pd.isna(position):
            return -1
        position = int(position)
        if position == 1:
            return 1 #Winner
        elif position <= 3: 
            return 2 #Podium - 15pts or more
        elif position <= 7:
            return 3  #Top 7 - 6pts or more
        elif position <= 10:
            return 4 #Top 10 - 1pts or more
        else:
            return 5 #Bottom 10 - 0pts 
    data_df["Session_5_position"] = data_df["Session_5_position"].apply(bin_finishing_position)
    return data_df

#Function to encode driver type to integers
def driver_type_encoding(data_df):
    data_df["Session_1_is_seat_driver"] = data_df["name_acronym"].isin(signed_drivers).astype(int)
    data_df["Session_2_is_seat_driver"] = data_df["name_acronym"].isin(signed_drivers).astype(int)
    data_df["Session_3_is_seat_driver"] = data_df["name_acronym"].isin(signed_drivers).astype(int)
    data_df["Session_4_is_seat_driver"] = data_df["name_acronym"].isin(signed_drivers).astype(int)
    data_df["Session_5_is_seat_driver"] = data_df["name_acronym"].isin(signed_drivers).astype(int)
    return data_df

#Function to impute environment features
def environment_features_imputation(data_df):
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
                    "Session_5_starting_pressure"]
    # Create a boolean mask for rows where 'circuit_key' is the same as the previous row
    is_same_circuit = data_df['circuit_key'] == data_df['circuit_key'].shift(1)

    for feature in environment_impute:
        # Check if the feature exists in the DataFrame
        if feature in data_df.columns:
            # Create a mask for nulls in the feature that should be forward-filled
            ffill_mask = data_df[feature].isnull() & is_same_circuit
            
            # Get the forward-filled values for the masked locations and assign them
            data_df.loc[ffill_mask, feature] = data_df[feature].ffill()[ffill_mask]
            
            # For the remaining nulls, impute with the value from the following entry (backward fill)
            data_df[feature].fillna(method='bfill', inplace=True)
    
    return data_df

#Function to impute position features
def position_features_imputation(data_df):
    position_features = ["Session_1_position", "Session_2_position", "Session_3_position", "Session_4_position"]

    for feature in position_features:
        # Check if the feature exists in the DataFrame
        if feature in data_df.columns:
            # Fill missing values with -1, indicating the driver did not participate or finish
            data_df[feature].fillna(-1, inplace=True)
    
    return data_df

def preprocessing(data_df):
    data_df = constructor_encoding(data_df)
    data_df = session_type_encoding(data_df)
    data_df = race_result_encoding(data_df)
    data_df = driver_type_encoding(data_df)
    data_df = environment_features_imputation(data_df)
    data_df = position_features_imputation(data_df)
    return data_df[feature_names], data_df["Session_5_position"]

