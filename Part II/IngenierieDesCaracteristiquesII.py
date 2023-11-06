import numpy as np
import pandas as pd
import math
from pandas import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from comet_ml import Experiment
from datetime import datetime


########################### Get data ############################
data = pd.read_csv('combined_data.csv')


### Calculer la distance ####
def calculate_distance(x1,y1, x2,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

### Calculer l'angle de tir #######
def calculate_angle(x, y, goal_x_positive=89, goal_x_negative=-89, goal_y=0):
    if x > 0:
        return math.degrees(math.atan2(goal_y - y, goal_x_positive - x))
    else:
        return math.degrees(math.atan2(goal_y - y, goal_x_negative - x))

#### Convertir le temps en secondes ###############
def time_to_seconds(time_str):
    # Assuming the format is "%M:%S" as placeholder
    mins, secs = map(int, time_str.split(":"))
    return mins * 60 + secs


######## Definir datetime #######
def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%SZ')



############# Initialiser les nouvelles colonnes ###############
# Initialize columns for new features
data['shot_distance'] = np.nan
data['shot_angle'] = np.nan
data['last_event_type'] = None
data['last_event_x'] = None
data['last_event_y'] = None
data['time_since_last_event'] = np.nan
data['distance_from_last_event'] = np.nan
data['is_rebound'] = False
data['shot_angle_change'] = 0
data['event_speed'] = 0


######################### Transform actual features into new features ############################

for i in range(len(data)):
    # Current event
    event = data.iloc[i]

    # Calculate shot distance and angle
    goal_y = 0
    goal_x = 89 if event['coordinates.x'] > 0 else -89
    data.at[i, 'shot_distance'] = calculate_distance(event['coordinates.x'], event['coordinates.y'], goal_x, goal_y)
    data.at[i, 'shot_angle'] = calculate_angle(event['coordinates.x'], event['coordinates.y'])

    if i > 0:
        # Previous event
        prev_event = data.iloc[i - 1]

        # Calculate time since the last event
        current_time = parse_datetime(event['about.dateTime'])
        last_time = parse_datetime(prev_event['about.dateTime'])
        data.at[i, 'time_since_last_event'] = (current_time - last_time).total_seconds()

        # Calculate distance from the last event
        data.at[i, 'distance_from_last_event'] = calculate_distance(prev_event['coordinates.x'],
                                                                  prev_event['coordinates.y'], event['coordinates.x'],
                                                                  event['coordinates.y'])

        # Check if the last event was a shot (for rebound calculation)
        data.at[i, 'is_rebound'] = prev_event['result.eventTypeId'] == 'SHOT'

        # Calculate shot angle change and event speed if the last event was a shot
        if data.at[i, 'is_rebound']:
            data.at[i, 'shot_angle_change'] = abs(data.at[i, 'shot_angle'] - data.at[i - 1, 'shot_angle'])
            data.at[i, 'event_speed'] = data.at[i, 'distance_from_last_event'] / data.at[i, 'time_since_last_event'] if data.at[
                                                                                                                      i, 'time_since_last_event'] > 0 else np.nan

        # Record last event type and coordinates
        data.at[i, 'last_event_type'] = prev_event['result.eventTypeId']
        data.at[i, 'last_event_x'] = prev_event['coordinates.x']
        data.at[i, 'last_event_y'] = prev_event['coordinates.y']

# Save the enhanced DataFrame to a new CSV file
enhanced_csv_path = "ingenierie_data.csv"
data.to_csv(enhanced_csv_path, index=False)
