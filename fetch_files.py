import os
import json

def fetch_data(directory_path):
# Spécifiez le chemin du répertoire contenant les fichiers crees precedement
    data_dir = directory_path

# Listez les fichiers dans le répertoire
    files = os.listdir(data_dir)

# Parcourez la liste des fichiers
    for filename in files:
    # Vérifiez si le fichier est au format JSON (ou tout autre format que vous utilisez)
        if filename.endswith(".json"):
        # Construisez le chemin complet du fichier
            file_path = os.path.join(directory_path, filename)
        
        # Lisez le contenu du fichier
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
    return data
#test de la fonction
#data= fetch_data("nhl_data/20162017")
#game=data["gameData"]["teams"]
#print(game)
