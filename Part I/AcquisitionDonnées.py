import os
import json
import requests
import pandas as pd

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
    data_dir = "nhl_data"  # Répertoire de stockage des données
    downloader = NHLPBPDownloader(data_dir)
    
    start_year = 2016
    end_year = 2021

    for year in range(start_year, end_year + 1):
        # Convertir l'année au format approprié, e.g., "20162017" for the 2016-17 saison
        season = f"{year}{year + 1}"
        # Téléchargez les données pour la saison en cours
        season_data = downloader.download_season_data(season)
