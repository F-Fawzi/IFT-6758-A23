import pandas as pd
import json
from pathlib import Path

# The directory where your JSON files are stored
json_dir = Path('nhl_penalty_data')

# Prepare an empty list for the combined data
combined_data = []

# Read each JSON file and extract the required information
for json_file in json_dir.glob('*.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)

        # Extract the game ID from the file name
        game_id = json_file.stem.split('_')[-1]

        # Extract 'penaltyPlays' and 'about' details
        if 'liveData' in data and 'plays' in data['liveData']:
            plays_data = data['liveData']['plays']
            penalty_plays_indices = plays_data.get('penaltyPlays', [])
            all_plays = plays_data.get('allPlays', [])

            # Iterate over the indices in 'penaltyPlays' to get the actual penalty plays
            for idx in penalty_plays_indices:
                penalty_play = all_plays[idx] if idx < len(all_plays) else None
                if penalty_play:
                    about = penalty_play.get('about', {})
                    period = about.get('period')
                    datetime = about.get('dateTime')

                    # Append the data to the list
                    combined_data.append({
                        'ID_game': game_id,
                        'about.period': period,
                        'about.dateTime': datetime,
                        'penaltyPlays': penalty_play
                    })

# Create a DataFrame from the combined data list
combined_df = pd.DataFrame(combined_data)

# Save the combined data to a CSV file
combined_df.to_csv('combined_penalty_plays.csv', index=False)



##############################################################################  RECUPERATION DES DONNEES DE PENALTY  ##############################################

import pandas as pd
import json
from datetime import datetime, timedelta


# Fonction pour convertir la chaîne de date et heure en objet datetime
import pandas as pd
from datetime import datetime, timedelta

# Charger le fichier CSV
df = pd.read_csv('combined_penalty_plays.csv')

# Préparation des fonctions de parsing
def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")

# Calcul du temps écoulé depuis le début du match
def calculate_elapsed_time(game_start, current_time, period_time_str):
    period_length = 20 * 60  # Chaque période est de 20 minutes
    mins, secs = map(int, period_time_str.split(":"))
    period_time_secs = mins * 60 + secs
    return (current_time - game_start).total_seconds() + period_time_secs

# Calcul du temps de jeu de puissance
def calculate_power_play_time(penalties, current_time):
    power_play_time = 0
    penalties = sorted(penalties, key=lambda x: x['end'])
    for penalty in penalties:
        if penalty['end'] > current_time:
            power_play_time += (penalty['end'] - current_time).total_seconds()
            current_time = penalty['end']
    return power_play_time

# Initialisation du dictionnaire pour suivre les jeux de puissance
game_power_plays = {}

# Traitement de chaque ligne du DataFrame
for index, row in df.iterrows():
    game_id = row['ID_game']
    period = row['about.period']
    current_time = parse_datetime(row['about.dateTime'])
    penalty_data = json.loads(row['penaltyPlays'].replace("'", '"'))
    period_time_str = penalty_data['about']['periodTime']
    penalty_end = current_time + timedelta(minutes=penalty_data['result']['penaltyMinutes'])

    if game_id not in game_power_plays:
        game_power_plays[game_id] = {
            'game_start': current_time,
            'penalties': [],
            'power_play_time_elapsed': 0
        }

    game_info = game_power_plays[game_id]
    game_start = game_info['game_start']
    elapsed_time = calculate_elapsed_time(game_start, current_time, period_time_str)
    penalties = game_info['penalties']
    penalties.append({'end': penalty_end, 'team': penalty_data['team']['name']})

    # Filtrer les pénalités terminées et trier par heure de fin
    penalties = [pen for pen in penalties if pen['end'] > current_time]
    power_play_time = calculate_power_play_time(penalties, current_time)

    # Réinitialisation du temps de jeu de puissance si nécessaire
    if not penalties:  # Toutes les pénalités sont terminées
        game_info['power_play_time_elapsed'] = 0
    else:
        game_info['power_play_time_elapsed'] += power_play_time

    # Mise à jour des données dans le DataFrame
    df.at[index, 'power_play_time_elapsed'] = game_info['power_play_time_elapsed']
    df.at[index, 'friendly_skaters'] = 6 - sum(pen['team'] == penalty_data['team']['name'] for pen in penalties)
    df.at[index, 'opposing_skaters'] = 6 - sum(pen['team'] != penalty_data['team']['name'] for pen in penalties)

# Sauvegarder les résultats
df.to_csv('power_plays.csv', index=False)


############################################################################ COMBINAISON CSV ###################################################################
import pandas as pd

# Charger les données depuis les fichiers CSV
df_ingenierie = pd.read_csv('ingenierie_data.csv')
df_power_plays = pd.read_csv('power_plays.csv')

# On peut supprimer la colonne 'Unnamed: 0' car elle semble être un index résiduel de l'exportation
df_ingenierie.drop('Unnamed: 0', axis=1, inplace=True)

# Fusionner les deux DataFrames sur les colonnes communes
combined_df = pd.merge(df_ingenierie, df_power_plays[['ID_game', 'about.period', 'about.dateTime', 'power_play_time_elapsed', 'friendly_skaters', 'opposing_skaters']],
                       on=['ID_game', 'about.period', 'about.dateTime'], how='left')

# Réorganiser les colonnes pour que ID_game soit la première
col_order = ['ID_game'] + [col for col in combined_df.columns if col != 'ID_game']
combined_df = combined_df[col_order]

# Enregistrer le résultat dans un nouveau fichier CSV
combined_df.to_csv('final_data.csv', index=False)




