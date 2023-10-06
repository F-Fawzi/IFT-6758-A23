import os
import re
import json
import pandas as pd
from pandas import json_normalize

# Define a regular expression pattern to match filenames

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

allplays_path=['liveData', 'plays', 'allPlays']

def fetch_data(data_dir): 
    # Iterate through the files in the directory
    for year in os.listdir(data_dir):
        for filename in os.listdir(year):

            if filename.endswith('.json'):
                name=filename.rsplit('.json')[0]
                print(name[-10:]) 

                # Construct the full path to the JSON file
                json_path = os.path.join(data_dir, filename)
                
                # Read the JSON file and directly convert it to a DataFrame
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                        # Extract the data from the desired nested dictionary
                nested_data = json_data
                for key in allplays_path:
                    nested_data = nested_data.get(key, {})
                
                df=json_normalize(nested_data, )

                # Add an additional column with the matched string
                df['ID_game'] = name


                # Concatenate the current DataFrame with the combined DataFrame
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Now 'combined_df' contains your data from all matching JSON files in a single Pandas DataFrame

if __name__ == "__main__":

    data_dir = "nhl_data" # Répertoire de stockage des données
    fetch_data(data_dir="nhl_data") 
    start_year = 2016
    end_year = 2021
