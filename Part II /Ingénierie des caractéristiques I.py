import os
import json
import requests
import pandas as pd
import re
from pandas import json_normalize

class NHLPBPDownloader:
    def __init__(self, data_dir):
        """
        Initializes an instance of NHLPBPDownloader.

        Args:
            data_dir (str): The directory where data will be stored.

        Attributes:
            base_url (str): The base URL for NHL data API.
            data_dir (str): The directory where data will be stored.

        """
        self.base_url = "https://statsapi.web.nhl.com/api/v1"
        self.data_dir = data_dir
    
    def download_season_data(self, season):
        """
        Downloads and structures NHL play-by-play data for a given season.

        Args:
            season (str): The NHL season in the format "YYYYYYYY" (e.g., "20162017").

        Returns:
            None: Data is downloaded and stored in the specified directory.

        """
        season_dir = os.path.join(self.data_dir, str(season))
        os.makedirs(season_dir, exist_ok=True)
        # Vérifiez si les données existent localement
        season_file = os.path.join(data_dir, f"nhl_data_{season}.json")        
        if os.path.exists(season_file):
            # Si les données existent, chargez-les depuis le fichier
            with open(season_file, "r") as json_file:
                return json.load(json_file)
        else:
            # Si les données n'existent pas localement, téléchargez-les depuis l'API REST
            # Récupérez la liste des jeux pour la saison régulière
            season_reguliere_url = f"{self.base_url}/schedule?season={season}&gameType=R"
            response = requests.get(season_reguliere_url)
            schedule_reguliere_data = response.json()

            # Récupérez la liste des jeux pour la saison éliminatoire
            saison_eliminatoire_url = f"{self.base_url}/schedule?season={season}&gameType=P"
            response = requests.get(saison_eliminatoire_url)
            schedule_eliminatoire_data = response.json()

            #Récuperation des données de saison régulière et éliminatoire
            # Récupération des données de saison régulière
            for game_date  in schedule_reguliere_data["dates"]:
                for game_info in game_date["games"]:
                    game_id = game_info["gamePk"]
                    play_by_play_url = f"{self.base_url}/game/{game_id}/feed/live/"
                    try:
                        response = requests.get(play_by_play_url)
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as err:
                        raise SystemExit(err)
                    play_by_play_data = response.json()
                    season = str(game_info["season"])  # Convert season to a string
                    # Define the directory path for the season
                    #season_dir = os.path.join(self.data_dir, season)
                    # Create the season directory if it doesn't exist
                    #os.makedirs(season_dir, exist_ok=True)
                    # Definir le path du fichier json pour enregistrer Data
                    game_file = os.path.join(season_dir, f"nhl_game_{game_id}.json")
                    # Define the directory path for the season
                    # Enregistrer Data dans json spécifique pour la saison réguliere
                    with open(game_file, "w") as json_file:
                        json.dump(play_by_play_data, json_file, indent=4)

            # Récupération des données de saison éliminatoire
            for game_date  in schedule_eliminatoire_data["dates"]:
                for game_info in game_date["games"]:
                    game_id = game_info["gamePk"]
                    play_by_play_url = f"{self.base_url}/game/{game_id}/feed/live/"
                    response = requests.get(play_by_play_url)
                    play_by_play_data = response.json()
                    # Definir le path du fichier json pour enregistrer Data
                    game_file = os.path.join(season_dir, f"nhl_game_{game_id}.json")
                    # Enregistrer Data dans json spécifique pour la saison éliminatoire
                    with open(game_file, "w") as json_file:
                            json.dump(play_by_play_data, json_file, indent=4)
       
        return 

if __name__ == "__main__":
    data_dir = "/Users/francis-olivierbeauchamp/Desktop/nhl_data/"  # Répertoire de stockage des données
    downloader = NHLPBPDownloader(data_dir)
    
    start_year = 2015
    end_year = 2020

    for year in range(start_year, end_year + 1):
        # Convertir l'année au format approprié, e.g., "20162017" for the 2016-17 saison
        season = f"{year}{year + 1}"
        # Téléchargez les données pour la saison en cours
        season_data = downloader.download_season_data(season)

allplays_path=['liveData', 'plays', 'allPlays']
players_path=['liveData', 'plays', 'allPlays', 'players']


def fetch_data(year_folders, start_year, end_year): 
    """
    Fetches and structures NHL play-by-play data from JSON files for the specified years and saves it as CSV.

    Args:
        year_folders (str): The path to the directory containing NHL data organized by seasons.
        start_year (int): The starting year for data retrieval.
        end_year (int): The ending year for data retrieval.

    Returns:
        None: The function saves the structured data as CSV files for each season.

    """
    start_year-=1

    # Iterate through the files in the directory
    for season in sorted(os.listdir(year_folders)): 
        combined_df = pd.DataFrame()
        start_year+=1
        season=os.path.join(year_folders, season)
        if season.endswith('.DS_Store'):

            os.remove(season)
            print(f'this {season} has been deleted')

        for filename in sorted(os.listdir(season)):
            print(filename)
            if filename.endswith('.json'):

                game_ID=filename.rsplit('.json')[0]
                game_ID=game_ID[-10:]
                if game_ID[-6:-4]== '02':
                    game_type= 'regular'
                if game_ID[-6:-4]== '03':
                    game_type= 'playoffs'

                

                # Construct the full path to the JSON file
                json_path = os.path.join(season, filename)
                
                # Read the JSON file and directly convert it to a DataFrame
                with open(json_path, 'r') as file:
                    json_data = json.load(file)
                        # Extract the data from the desired nested dictionary
                nested_data = json_data
                for key in allplays_path:
                    nested_data = nested_data.get(key, {})
                
                for play in nested_data:
           
                    try :
                        scorer=play["players"][0]
                        goalie=play["players"][-1]
                        scorer=json_normalize(scorer)
                        scorer=scorer.add_suffix('_Scorer')
                        goalie=json_normalize(goalie)
                        goalie=goalie.add_suffix('_goalie')
                        player_col=pd.concat([scorer,goalie], axis=1)
              
                        if play["result"]["eventTypeId"]=="SHOT" or play["result"]["eventTypeId"]=="GOAL":

                            df=json_normalize(play)
                            df=pd.concat([player_col,df], axis=1)
                            df['ID_game'] = game_ID
                            df['Game_type']= game_type

                            # Concatenate the current DataFrame with the combined DataFrame
                            combined_df = pd.concat([combined_df, df])

                    except Exception as e: 
                        continue

    # Now 'combined_df' contains your data from all matching JSON files in a single Pandas DataFrame
        combined_df=combined_df[["about.period","about.dateTime","team.name","result.eventTypeId","coordinates.x","coordinates.y","result.secondaryType","result.emptyNet","result.strength.name","player.fullName_Scorer","player.fullName_goalie","ID_game","Game_type"]]
        combined_df.to_csv(f"{start_year}.csv")


if __name__ == "__main__":

    year_dir = "/Users/francis-olivierbeauchamp/Desktop/nhl_data" # Répertoire de stockage des données
    fetch_data(year_dir, start_year=2015, end_year=2020)

# Directory path where your CSV files are located
directory = "/Users/francis-olivierbeauchamp/Desktop/nhl_data"

# Initialize an empty DataFrame
combined_data = pd.DataFrame()

# Loop through the years 2015 to 2020
for year in range(2015, 2021):
    file_name = f"{year}.csv"
    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    else:
        print(f"File not found for {year}: {file_name}")

# Save the combined DataFrame to a new CSV file
combined_data.to_csv("combined_data.csv", index=False)

#### Après le dowload des bonnes saisons avec la technique utilisée dans la partie 1 du projet :::

# Import important stuff
import csv
import pandas as pd
import numpy as np 
from scipy.spatial import distance 
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Open file
csv_file_path = "/Users/francis-olivierbeauchamp/Desktop/nhl_data/combined_data.csv"

# Open the CSV file in read mode
with open(csv_file_path, mode='r') as file:
    csv_reader = csv.reader(file)

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Define the coordinates of the two goals
goal_1 = (89, 0)
goal_2 = (-89, 0)

def determine_target_goal(data):
    team_shots = data['coordinates.x'].mean()
    
    # Determine the target goal based on the average coordinates.x
    if team_shots > 0:
        return 1  # Team primarily shoots in positive coordinates.x
    else:
        return 2  # Team primarily shoots in negative coordinates.x

# Calculate the target goal for each period and team for every 'ID_game'
target_goals = data.groupby(['ID_game', 'about.period', 'team.name']).apply(determine_target_goal).reset_index(name='target_goal')

# Merge the target goal information with the main data
data = data.merge(target_goals, on=['ID_game', 'about.period', 'team.name'])


# Create a function to calculate Euclidean distance
def calculate_distance(x, y, target_goal):
    dx = target_goal[0] - x
    dy = target_goal[1] - y
    return math.sqrt(dx**2 + dy**2)

# Create a function to calculate the angle in degrees relative to the goal
def calculate_angle(x, y, target_goal):
    if target_goal == goal_1:
        target_x = 89
    else:
        target_x = -89
    
    dx = target_x - x
    dy = 0 - y  

    if dy == 0:
        # For the case where the point is on the vertical line (dy is zero)
        angle = 90  # Set to 90 degrees
    else:
        angle = math.degrees(math.asin(dx / math.sqrt(dx**2 + dy**2)))
    
    # Ensure the angle is in the range [0, 360]
    angle -= 360 if angle > 90 else 0
    angle = abs(angle)  # Ensure the angle is positive
    return angle

# Convert 'False' to 0, 'True' to 1, and 'NaN' to 0 in the 'result.emptyNet' column
data["empty_net"] = data["result.emptyNet"].map({False: 0, True: 1}).fillna(0)

# Calculate the Euclidean distance and angle for each shot and goal
data["distance_to_target_goal"] = data.apply(
    lambda row: calculate_distance(row["coordinates.x"], row["coordinates.y"], goal_1 if row["target_goal"] == 1 else goal_2), 
    axis=1
)
data["angle_to_target_goal"] = data.apply(
    lambda row: calculate_angle(row["coordinates.x"], row["coordinates.y"], goal_1 if row["target_goal"] == 1 else goal_2), 
    axis=1
)

# Set a column for shot/goal (0 for shot, 1 for goal)
data["shot_or_goal"] = data["result.eventTypeId"].apply(lambda x: 1 if x == "GOAL" else 0)

# Create the training dataset with the desired columns
df_combined = data[["distance_to_target_goal", "angle_to_target_goal", "shot_or_goal", "empty_net"]]

# Filter data for shots and goals (excluding empty net goals as it gives a better understanding of the dynamics)
filtered_data = data[(data["empty_net"] == 0.0)]

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a histogram for shots
sns.histplot(filtered_data[filtered_data["shot_or_goal"] == 0]["distance_to_target_goal"], kde=True, color="blue", label="Shots", bins=20)

# Create a histogram for goals
sns.histplot(filtered_data[filtered_data["shot_or_goal"] == 1]["distance_to_target_goal"], kde=True, color="red", label="Goals", bins=20)
# Set labels and title
plt.xlabel("Distance to Target Goal")
plt.ylabel("Count")
plt.title("Histogram of Shots and Goals (Excluding Empty Net) by Distance")
# Add a legend
plt.legend()
# Show the plot
plt.show()

# Separate data into shots and goals
shots = data[data['shot_or_goal'] == 0]
goals = data[data['shot_or_goal'] == 1]

# Set up the plot
plt.figure(figsize=(10, 6))
# Create a histogram for shots and goals binned by angle using Seaborn
sns.histplot(shots['angle_to_target_goal'], color='blue', label='Shots', kde=True)
sns.histplot(goals['angle_to_target_goal'], color='red', label='Goals', kde=True)
# Set labels and title
plt.xlabel("Angle to Target Goal")
plt.ylabel("Total Count")
plt.title("Histogram of Events (Shots and Goals) by Angle")
# Add a legend
plt.legend()
# Show the plot
plt.show()

# Create a jointplot
sns.jointplot(x='distance_to_target_goal', y='angle_to_target_goal', data=data, kind='hist', color='blue')
# Set labels and title
plt.xlabel("Distance to Target Goal")
plt.ylabel("Angle to Target Goal")
plt.suptitle("2D Histogram of Distance and Angle")
# Show the plot
plt.show()

# Create custom distance bins
bins = np.arange(0, 201, 20)

# Use pd.cut to assign each data point to a distance bin
data['distance_bin'] = pd.cut(data['distance_to_target_goal'], bins=bins, right=False)

# Calculate the goal rate in each distance bin
goal_rates = data.groupby('distance_bin').apply(lambda x: (x['shot_or_goal'] == 1).mean())

# Create a bar plot to compare goal rates in distance bins
plt.figure(figsize=(10, 6))
sns.barplot(x=goal_rates.index, y=goal_rates.values)
plt.xlabel("Distance Bin (in meters)")
plt.ylabel("Goal Rate")
plt.title("Goal Rate by Distance Bin")
plt.xticks(rotation=0)
plt.show()

# Create custom distance bins
bins = np.arange(0, 201, 20)

# Use pd.cut to assign each data point to a distance bin
data['distance_bin'] = pd.cut(data['distance_to_target_goal'], bins=bins, right=False)

# Filter data for shots and goals (excluding empty net goals as it gives a more representative understanding)
filtered_data = data[(data["empty_net"] == 0.0)]

# Calculate the goal rate in each distance bin for filtered data
goal_rates = filtered_data.groupby('distance_bin').apply(lambda x: (x['shot_or_goal'] == 1).mean())

# Create a bar plot to compare goal rates in distance bins
plt.figure(figsize=(10, 6))
sns.barplot(x=goal_rates.index, y=goal_rates.values)
plt.xlabel("Distance Bin (in meters)")
plt.ylabel("Goal Rate")
plt.title("Goal Rate by Distance Bin (Excluding Empty Net)")
plt.xticks(rotation=0)
plt.show()

# Create custom angle bins
bins = np.arange(0, 100, 10)

# Use pd.cut to assign each data point to an angle bin
data['angle_bin'] = pd.cut(data['angle_to_target_goal'], bins=bins, right=False)

# Calculate the goal rate in each angle bin
goal_rates = data.groupby('angle_bin').apply(lambda x: (x['shot_or_goal'] == 1).mean())

# Create a bar plot to compare goal rates in angle bins
plt.figure(figsize=(10, 6))
sns.barplot(x=goal_rates.index, y=goal_rates.values)
plt.xlabel("Angle Bin (in degrees)")
plt.ylabel("Goal Rate")
plt.title("Goal Rate by Angle Bin")
plt.xticks(rotation=0)
plt.show()

# Filter data for goals with empty_net == 0
goals_empty_net_0 = data[(data['shot_or_goal'] == 1) & (data['empty_net'] == 0)]

# Filter data for goals with empty_net == 1
goals_empty_net_1 = data[(data['shot_or_goal'] == 1) & (data['empty_net'] == 1)]

# Set up the plot
plt.figure(figsize=(10, 6))

# Create a histogram for goals with empty_net == 0
sns.histplot(goals_empty_net_0["distance_to_target_goal"], kde=True, color="blue", label="Goals (Non-Empty Net)", bins=30)

# Create a histogram for goals with empty_net == 1
sns.histplot(goals_empty_net_1["distance_to_target_goal"], kde=True, color="red", label="Goals (Empty Net)", bins=30)

# Set labels and title
plt.xlabel("Distance to Target Goal (in meters)")
plt.ylabel("Goal Count (log scale)")
plt.title("Histogram of Goals by Distance (Empty Net vs. Non-Empty Net)")

# Add a legend
plt.legend()
# Use a logarithmic scale on the y-axis
plt.yscale('log')
# Show the plot
plt.show()
