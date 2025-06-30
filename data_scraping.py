# # Pulling data from Open F1 API

# Getting data

from urllib.request import urlopen
import json
import pandas as pd
import time
def get_openf1_data(endpoint, params=None):
    """
    Fetch data from the OpenF1 API.

    :param endpoint: The API endpoint to query.
    :param params: Optional dictionary of query parameters.
    :return: Parsed JSON response from the API.
    """
    base_url = 'https://api.openf1.org/v1/'
    params = '&'.join(f"{key}={value}" for key, value in (params or {}).items())
    url = f"{base_url}{endpoint}?{params}"
    try:
        response = urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(10)
        print("Retrying...")
        response = urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        return data
        #eturn {"error": str(e)}


print(get_openf1_data('car_data', {'driver_number': 55, 'session_key': 9159, 'speed': 315}))

def get_session_data(session):
    """
    Fetch following driver data for a specific session.
    Final position
    Starting weather conditions
    Ending weather conditions
    """
    session_key = session['session_key']
    session_type = session['session_type']
    session_name = session['session_name']
    circuit_short_name = session['circuit_short_name']
    date_start = session['date_start']
    date_end = session['date_end']
    drivers_data = get_openf1_data('drivers', {'session_key': session_key})
    
    if not drivers_data:
        print("Could not fetch driver data.")
        return None
    
    drivers_df = pd.DataFrame(drivers_data)

    drivers = drivers_df['driver_number'].unique()
    final_positions = pd.DataFrame(columns=['driver_number', 'position'])

    positions_data = get_openf1_data('position', {'session_key': session_key})

    positions_df = pd.DataFrame(positions_data)
    for driver in drivers:
        driver_positions = positions_df[positions_df['driver_number'] == driver]
        final_position = driver_positions.iloc[-1] if not driver_positions.empty else None
        final_positions = pd.concat([final_positions, pd.DataFrame({'driver_number': [driver], 'position': [final_position['position'] if final_position is not None else None]})], ignore_index=True)

    data_df = drivers_df.merge(final_positions, on='driver_number', how='left')
    data_df['session_type'] = session_type
    data_df['session_name'] = session_name
    data_df['circuit_short_name'] = circuit_short_name
    data_df['date_start'] = date_start
    data_df['date_end'] = date_end

    weather_data = get_openf1_data('weather', {'session_key': session_key})
    
    weather_df = pd.DataFrame(weather_data)
    weather_df = weather_df.sort_values('date')
    
    starting_weather = weather_df.iloc[0] if not weather_df.empty else None
    ending_weather = weather_df.iloc[-1] if not weather_df.empty else None

    starting_wind_direction = starting_weather['wind_direction'] if starting_weather is not None else None
    starting_wind_speed = starting_weather['wind_speed'] if starting_weather is not None else None
    starting_rainfall = starting_weather['rainfall'] if starting_weather is not None else None
    starting_track_temperature = starting_weather['track_temperature'] if starting_weather is not None else None
    starting_air_temperature = starting_weather['air_temperature'] if starting_weather is not None else None
    starting_humidity = starting_weather['humidity'] if starting_weather is not None else None
    starting_pressure = starting_weather['pressure'] if starting_weather is not None else None

    ending_wind_direction = ending_weather['wind_direction'] if ending_weather is not None else None
    ending_wind_speed = ending_weather['wind_speed'] if ending_weather is not None else None
    ending_rainfall = ending_weather['rainfall'] if ending_weather is not None else None
    ending_track_temperature = ending_weather['track_temperature'] if ending_weather is not None else None
    ending_air_temperature = ending_weather['air_temperature'] if ending_weather is not None else None
    ending_humidity = ending_weather['humidity'] if ending_weather is not None else None
    ending_pressure = ending_weather['pressure'] if ending_weather is not None else None

    data_df['starting_wind_direction'] = starting_wind_direction
    data_df['starting_wind_speed'] = starting_wind_speed
    data_df['starting_rainfall'] = starting_rainfall
    data_df['starting_track_temperature'] = starting_track_temperature
    data_df['starting_air_temperature'] = starting_air_temperature
    data_df['starting_humidity'] = starting_humidity
    data_df['starting_pressure'] = starting_pressure

    data_df['ending_wind_direction'] = ending_wind_direction
    data_df['ending_wind_speed'] = ending_wind_speed
    data_df['ending_rainfall'] = ending_rainfall
    data_df['ending_track_temperature'] = ending_track_temperature
    data_df['ending_air_temperature'] = ending_air_temperature
    data_df['ending_humidity'] = ending_humidity
    data_df['ending_pressure'] = ending_pressure

    print(f"Session key: {session_key}")
    print(f"Session type: {session_type}")
    res = data_df[["driver_number", "session_key", "session_type", "session_name", "date_start", "date_end", "starting_wind_direction", "starting_wind_speed", "starting_rainfall", "starting_track_temperature", "starting_air_temperature", "starting_humidity", "starting_pressure", "ending_wind_direction", "ending_wind_speed", "ending_rainfall", "ending_track_temperature", "ending_air_temperature", "ending_humidity", "ending_pressure", "position"]]
    return res

def get_meetings(date_start, date_end = None):
    if date_end is None:
        data = pd.DataFrame(get_openf1_data('meetings', {'date_start>': date_start}))
    else:
        data = pd.DataFrame(get_openf1_data('meetings', {'date_start>': date_start, 'date_end<': date_end}))
    return data

def get_sessions(meeting_key):
    data = pd.DataFrame(get_openf1_data('sessions', {'meeting_key': meeting_key}))
    return data

    
def get_meeting_data(meeting):
    meeting_key = meeting["meeting_key"]
    circuit_key = meeting["circuit_key"]
    meeting_date = meeting["date_start"]
    meeting_name = meeting["meeting_name"]
    drivers_data = get_openf1_data('drivers', {'meeting_key': meeting_key})
    
    if not drivers_data:
        print("Could not fetch driver data.")
        return None
    
    driver_df = pd.DataFrame(drivers_data)
    driver_df["circuit_key"] = circuit_key
    driver_df["meeting_start_date"] = meeting_date
    driver_df["meeting_name"] = meeting_name
    driver_df = driver_df.drop_duplicates(subset=["driver_number"])
    driver_df = driver_df[["driver_number", "meeting_key", "circuit_key", "meeting_name", "meeting_start_date", "full_name", "name_acronym", "team_name", "headshot_url"]]
    driver_df = driver_df.sort_values(by=["driver_number"])
    sessions_df = get_sessions(meeting_key)
    race = 0
    sprint = 0
    preseason = 0
    if "Pre-Season" in meeting_name:
        preseason = 1
    for i in range(5):
        session_name = "Session_" + str(i + 1)
        if i < sessions_df.shape[0]:
            session = sessions_df.iloc[i]
            session_data = get_session_data(session)
            if "Race" in session_name:
                race = 1
            if "Sprint" in session_name:
                sprint = 1
            print(session_name)
            # Rename session_data columns before merging to avoid conflicts
            session_data_renamed = session_data.rename(columns={
                'session_key': f'{session_name}_session_key',
                'session_type': f'{session_name}_session_type',
                'session_name': f'{session_name}_session_name',
                'date_start': f'{session_name}_date_start',
                'date_end': f'{session_name}_date_end',
                'starting_wind_direction': f'{session_name}_starting_wind_direction',
                'starting_wind_speed': f'{session_name}_starting_wind_speed',
                'starting_rainfall': f'{session_name}_starting_rainfall',
                'starting_track_temperature': f'{session_name}_starting_track_temperature',
                'starting_air_temperature': f'{session_name}_starting_air_temperature',
                'starting_humidity': f'{session_name}_starting_humidity',
                'starting_pressure': f'{session_name}_starting_pressure',
                'ending_wind_direction': f'{session_name}_ending_wind_direction',
                'ending_wind_speed': f'{session_name}_ending_wind_speed',
                'ending_rainfall': f'{session_name}_ending_rainfall',
                'ending_track_temperature': f'{session_name}_ending_track_temperature',
                'ending_air_temperature': f'{session_name}_ending_air_temperature',
                'ending_humidity': f'{session_name}_ending_humidity',
                'ending_pressure': f'{session_name}_ending_pressure',
                'position': f'{session_name}_position'
            })
            # Drop columns from session_data_renamed that already exist in driver_df except 'driver_number'
            cols_to_merge = [col for col in session_data_renamed.columns if col not in driver_df.columns or col == 'driver_number']
            driver_df = driver_df.merge(session_data_renamed[cols_to_merge], on='driver_number', how='left')
            driver_df = driver_df.sort_values(by=[f'{session_name}_position'])
        else:
            driver_df[f'{session_name}_session_key'] = None
            driver_df[f'{session_name}_session_type'] = None
            driver_df[f'{session_name}_session_name'] = None
            driver_df[f'{session_name}_date_start'] = None
            driver_df[f'{session_name}_date_end'] = None
            driver_df[f'{session_name}_starting_wind_direction'] = None
            driver_df[f'{session_name}_starting_wind_speed'] = None
            driver_df[f'{session_name}_starting_rainfall'] = None
            driver_df[f'{session_name}_starting_track_temperature'] = None
            driver_df[f'{session_name}_starting_air_temperature'] = None
            driver_df[f'{session_name}_starting_humidity'] = None
            driver_df[f'{session_name}_starting_pressure'] = None
            driver_df[f'{session_name}_ending_wind_direction'] = None
            driver_df[f'{session_name}_ending_wind_speed'] = None
            driver_df[f'{session_name}_ending_rainfall'] = None
            driver_df[f'{session_name}_ending_track_temperature'] = None
            driver_df[f'{session_name}_ending_air_temperature'] = None
            driver_df[f'{session_name}_ending_humidity'] = None
            driver_df[f'{session_name}_ending_pressure'] = None
            driver_df[f'{session_name}_position'] = None
        #time.sleep(5)
            

    if sessions_df.empty:
        print("No sessions found for this meeting.")
        return driver_df

    driver_df["Sprint Wknd"] = sprint
    driver_df["Race Wknd"] = race
    driver_df["Preseason Wknd"] = preseason
    return driver_df

meetings = get_meetings('2022-01-01', '2025-06-03')
data = get_meeting_data(meetings.iloc[0])
for i in range(1, meetings.shape[0]):
    meeting = meetings.iloc[i]
    meeting_data = get_meeting_data(meeting)
    data = pd.concat([data, meeting_data], ignore_index=True)

data = data.sort_values(by=["meeting_start_date", "circuit_key", "driver_number"])
data = data.reset_index(drop=True)
data.to_csv('openf1_data.csv', index=False)
print("Data saved to openf1_data.csv")

current_meetings = get_meetings('2025-06-03')
current_data = get_meeting_data(current_meetings.iloc[0])
for i in range(1, current_meetings.shape[0]):
    meeting = current_meetings.iloc[i]
    meeting_data = get_meeting_data(meeting)
    current_data = pd.concat([current_data, meeting_data], ignore_index=True)

current_data.to_csv('new_data.csv', index=False)
print("Current data saved to new_data.csv")

