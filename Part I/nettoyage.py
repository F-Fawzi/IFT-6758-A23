import os
import re
import json
import pandas as pd
from pandas import json_normalize

# Define a regular expression pattern to match filenames

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

allplays_path=['liveData', 'plays', 'allPlays']

def fetch_data(year_dir): 
    # Iterate through the files in the directory

    for filename in os.listdir(year_dir):

        if filename.endswith('.json'):

            game_ID=filename.rsplit('.json')[0]


            print(game_ID[-10:]) 

            # Construct the full path to the JSON file
            json_path = os.path.join(year_dir, filename)
            
            # Read the JSON file and directly convert it to a DataFrame
            with open(json_path, 'r') as file:
                json_data = json.load(file)
                    # Extract the data from the desired nested dictionary
            nested_data = json_data
            for key in allplays_path:
                nested_data = nested_data.get(key, {})
            
            for play in nested_data: 
                if play["result"]["eventTypeId"] is "SHOT" or "GOAL":
                    print(play["result"]["eventTypeId"])
                    df=json_normalize(play)
                    # Concatenate the current DataFrame with the combined DataFrame
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

                
            # Add an additional column with the matched string
            df['ID_game'] = game_ID


        # Now 'combined_df' contains your data from all matching JSON files in a single Pandas DataFrame

if __name__ == "__main__":

    year_dir = "test_data" # Répertoire de stockage des données
    fetch_data(year_dir) 


    start_year = 2016
    end_year = 2021
