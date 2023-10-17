---
layout: post
title: Milestone 1
---

## 

**Download NHL Data**


Nous allons créer une classe appelée NHLPBPDownloader pour gérer le processus de téléchargement des données. Voici comment l'initialiser :
```python
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
    
```

Nous allons créer une méthode qui permet de télécharger les données d'une saison spécifique. Dans cette méthode, vous interagirez avec l'API NHL Stats pour récupérer les calendriers des matchs, puis vous téléchargerez les données play-by-play pour chaque match de la saison. Nous ajouterons également du code pour enregistrer localement les données téléchargées afin d'éviter de les retélécharger à l'avenir.


```python
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
```


## Outil de débogage interactif

Nous avons ensuite nettoyé les données, afin de les transformées en pandas DataFrame, afin de pouvoir mieux les manipuler. 

Voici l'image de sortie générée à l'aide de l'outil interactif qui affiche la sortie de l'outil pour les choix suivants : "Type de match" ("regular") ; "Saison" ("2016") ; et pour "ID du match" ("0001"). Elle affiche également les statistiques pertinentes pour les options choisies.

![ice_rink](../_assets/images/widget.png)

Le code est ci-dessous :

```python
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import json
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ipywidgets import widgets
from IPython.display import display, Image, clear_output
from ipywidgets import interact, interactive, fixed, interact_manual

#Folder of a given season 
file_path='/Users/ceciliaacosta/IFT-DATASCIENCE/nhl_data/20172018'
#Path to the rink image
rink_image_path='/Users/ceciliaacosta/IFT-DATASCIENCE/MILESTONE1_local/nhl_rink.png'
year=file_path[-8:-4]

# Get the total number and the ID's to use in the slider (LAST FOUR DIGITS) of regular season and playoff games
regular_games=[filename for filename in (os.listdir(file_path)) if filename[-11:-9]=='02']
total_regular_season_games = len(regular_games)
idx_regular_games=sorted([int(id.rstrip('.json')[-4:]) for id in regular_games])

playoff_games = [filename for filename in (os.listdir(file_path)) if  filename[-11:-9]=='03']
total_playoff_games=len(playoff_games)
idx_playoff_games=sorted([int(id.rstrip('.json')[-3:]) for id in playoff_games])


def get_coordinates(nested_data, event_id):
    
    x=nested_data[event_id].get('coordinates').get('x',None)
    y=nested_data[event_id].get('coordinates').get('y',None)

    return x, y 

#load data
def load_data(game_ID, season):
    global file_path
    if season=='regular_season':
        season_code='02'
    else:
        season_code='03'

    if len(str(game_ID))==1:
        files_game = f"{file_path}/nhl_game_{year}{season_code}000{game_ID}.json"
    if len(str(game_ID))==2:
        files_game = f"{file_path}/nhl_game_{year}{season_code}00{game_ID}.json"
    if len(str(game_ID))==3:
        files_game = f"{file_path}/nhl_game_{year}{season_code}0{game_ID}.json"
    if len(str(game_ID))==4:
        files_game = f"{file_path}/nhl_game_{year}{season_code}{game_ID}.json"

    if os.path.exists(files_game):

        with open(files_game, 'r') as file:
            return json.load(file)
    else:
        return None

output = widgets.Output()
def display_game_events(game_data):

    allplays_path=['liveData', 'plays', 'allPlays']
    nested_data = game_data

    #Update the event ID slider (aka the index in the list of events)
    for key in allplays_path:
        nested_data = nested_data.get(key, {})
    event_id_slider.max=len(nested_data)-1
    event_id = event_id_slider.value
    # with output:
    #     clear_output(wait=True)
    clear_output(wait=True)

    if game_data:
        x_coord, y_coord=get_coordinates(nested_data, event_id)
        if x_coord== None and y_coord==None:
            pprint(nested_data[event_id])
            event_type=nested_data[event_id].get('result',{}).get('description',"")
        else:
            event_type=nested_data[event_id].get('result',{}).get('description',"")
# Display event information on the ice rink image
            # Load the rink image to get its dimensions
            rink_image = mpimg.imread(rink_image_path)
            fig, ax = plt.subplots()
            ax.patch.set_facecolor('black')
            im = ax.imshow(rink_image, origin='lower', extent=[-100, 100, -42, 42])
            ax.plot(x_coord,y_coord, 'bo')
            ax.set_title(f'{event_type}')
            ax.set_xlabel('feet')
            ax.set_ylabel('feet')
            plt.show()
            pprint(nested_data[event_id])

    # Function to update the display based on the selected season and game ID
def update_display(change):
    season = 'regular_season' if season_dropdown.value == 'Regular Season' else 'playoffs'
    game_id = game_id_slider.value
    game_data = load_data(game_id, season)

    if game_data:
        display_game_events(game_data)

    else:

        print(f"Game data not found for Game ID {game_id} in the {season}.")


# Attach the update_display function to the widgets' observe methods
# Create a dropdown widget to select the season
season_dropdown = widgets.Dropdown(
    options=['Regular Season', 'Playoffs'],
    description='Season:',
)

#Create an IntSlider for Game ID with a maximum value that depends on the selected season
game_id_slider = widgets.IntSlider(
    min=1,
    max=total_regular_season_games,
    description='Game ID:',
    continuous_update=False,
    options=idx_regular_games
)

event_id_slider = widgets.IntSlider(
    min=0,
    max=0,  # Initialize with 0 events
    description='Event ID:',
    continuous_update=False,
)

#Function to update the slider based on the selected type of Season 
def update_slider(change):
    if season_dropdown.value == 'Regular Season':
        game_id_slider.max = max(idx_regular_games)
        game_id_slider.min = min(idx_regular_games)
        game_id_slider.options=idx_regular_games

    else:
        game_id_slider.max = max(idx_playoff_games)
        game_id_slider.min= min(idx_playoff_games)
        game_id_slider.options=idx_playoff_games

# Attach the update_display function to the widgets' observe methods
# Create a dropdown widget to select the season
season_dropdown = widgets.Dropdown(
    options=['Regular Season', 'Playoffs'],
    description='Season:',
)

season_dropdown.observe(update_slider, 'value')
game_id_slider.observe(update_display, 'value')
event_id_slider.observe(update_display, 'value')

display(output)
update_display(None) 

# Display the widgets
season_dropdown.observe(update_slider, 'value')
game_id_slider.observe(update_display, 'value')
event_id_slider.observe(update_display, 'value')
display(season_dropdown, game_id_slider, event_id_slider)

```
## Task 4 - Nettoyage des données


**Answer 4.1**<br>



**Answer 4.2**<br>


<img src="../_assets/image-2.png" alt="penalty-reference" width="400"/>


**Answer 4.3**<br>


## Task 5 - Simple Visualization
**Answer 5.1**<br>

<img src="../_assets/image-4.png" alt="Sime-viz-1-table" width="1000"/>

![Simple-viz-1](../_assets/image-3.png)


### Analysis:



**Answer 5.2**<br>
![Simple-viz-2](../_assets/image-6.png)




**Answer 5.3**<br>
![Simple-viz-3](../_assets/image-7.png)


## Task 6 - Advanced Visualizations: Shot Maps

### Advanced Visualization Notes:


#### Analysis Assumptions 


```python

```

```python

```

<img src="../../data/excess_shot_rate_example.png" alt="example" width="1000"/>

#### How a single figure is plotted? 


**5.1 - 4 plot offensive zone plots visualization**


[Advanced Visualization Shot Map Plot](https://nhlhockeyapp.onrender.com/)

Furthermore, we are providing our HTML plot for your reference:
<iframe src="https://nhlhockeyapp.onrender.com/" title="Advanced Visualizations - Shot Maps" width="990" height="620"> </iframe>

Here is a concise summary of the logs when we deployed our application on Render:

![Deployment Logs](../_assets/images/Deploy_logs.png)

**5.2 - Plot interpretation**


**5.3 - Discussion on Performance Difference of Colorado Avalanche**


![Colorado Avalanche1](../_assets/images/2016_17_Colorado_Avalanche_Team.png)


![Colorado Avalanche2](../_assets/images/2020_21_Colorado_Avalanche_Team.png)



**5.4 - Performance Comparison between Buffalo Sabres and Tampa Bay Lightning**


Graphs depicting the performance of the Tampa Bay Lightning in the seasons 2018-19, 2019-20, and 2020-21:-

![Tampa Bay Lightning1](../_assets/images/2018_19_Tampa_Bay_Lightning.png)
![Tampa Bay Lightning2](../_assets/images/2019_20_Tampa_Bay_Ligtning.png)
![Tampa Bay Lightning3](../_assets/images/2020_21_Tampa_Bay_Lightning.png)

Plots illustrating the performance of the Buffalo Sabres in the 2018-19, 2019-20, and 2020-21 seasons:-

![Buffalo Sabres1](../_assets/images/2018_19_Buffalo_Sabres_Team.png)
![Buffalo Sabres2](../_assets/images/2019_20_Buffalo_Sabres_Team.png)
![Buffalo Sabres3](../_assets/images/2020_21_Buffalo_Sabres_Team.png)

