import requests
import json

import pandas as pd

# Remplacez 'nom_du_fichier.json' par le nom de votre fichier JSON
df = pd.read_json('nhl_data/nhl_data_20162017.json')
#print(df.columns)

colonne = df['liveData']

# Afficher la colonne
print(colonne)