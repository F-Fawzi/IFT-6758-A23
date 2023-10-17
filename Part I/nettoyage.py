import os
import re
import json
import pandas as pd
from pandas import json_normalize


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

    year_dir = "/Users/ceciliaacosta/IFT-DATASCIENCE/nhl_data" # Répertoire de stockage des données
    fetch_data(year_dir, start_year=2016, end_year=2021)
