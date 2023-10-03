import os
import json
import requests
import pandas as pd

class NHLPBPDownloader:
    def __init__(self, data_dir):
        self.base_url = "https://statsapi.web.nhl.com/api/v1"
        self.data_dir = data_dir
    
    def download_season_data(self, season):
        # Vérifiez si les données existent localement
        season_file = os.path.join(self.data_dir, f"nhl_data_{season}.json")        
        if os.path.exists(season_file):
            # Si les données existent, chargez-les depuis le fichier
            with open(season_file, "r") as json_file:
                return json.load(json_file)
        else:
            df = pd.DataFrame()
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
            for game in schedule_reguliere_data["dates"]:
                game_id = game["games"][0]["gamePk"]
                play_by_play_url = f"{self.base_url}/game/{game_id}/feed/live/"
                response = requests.get(play_by_play_url)
                play_by_play_data = response.json()
            # Definir le path du fichier json pour enregistrer Data
            game_file = os.path.join(self.data_dir, f"nhl_game_{game_id}.json")
            # Enregistrer Data dans json spécifique pour la saison réguliere
            with open(game_file, "w") as json_file:
                json.dump(play_by_play_data, json_file, indent=4)

            # Récupération des données de saison éliminatoire
            for game in schedule_eliminatoire_data["dates"]:
                game_id = game["games"][0]["gamePk"]
                play_by_play_url = f"{self.base_url}/game/{game_id}/feed/live/"
                response = requests.get(play_by_play_url)
                play_by_play_data = response.json()
            # Definir le path du fichier json pour enregistrer Data
            game_file = os.path.join(self.data_dir, f"nhl_game_{game_id}.json")
            # Enregistrer Data dans json spécifique pour la saison éliminatoire
            with open(game_file, "w") as json_file:
                   json.dump(play_by_play_data, json_file, indent=4)
       
        return 

if __name__ == "__main__":
    data_dir = "nhl_data"  # Répertoire de stockage des données
    downloader = NHLPBPDownloader(data_dir)
    
    # Téléchargez les données pour la saison 2016-17
    season_data = downloader.download_season_data("20172018")
