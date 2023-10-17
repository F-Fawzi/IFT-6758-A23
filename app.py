#Import library
from base64 import encode
import plotly.express as px
from plotly.figure_factory import create_quiver
import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Define the rink dimensions 
rink_width_feet = 85  # Width of the rink in feet
rink_length_feet = 200  # Length of the rink in feet

def shotrate_per_location(df):
    # Calculate the league average shot rate per hour across all locations
    total_games = df['ID_game'].nunique()

    # Total shots per location 
    ser_total_shots= df.groupby(['coordinates.x', 'coordinates.y']).count()
    df_total_shots=pd.DataFrame(ser_total_shots['ID_game'])
    df_total_shots=df_total_shots.rename(columns={'ID_game':'coord_total_shots'})

    # Calculate the rate per hour of the shots per location
    df_rate_shots=df_total_shots.copy()
    df_rate_shots['coord_total_shots'] = df_rate_shots['coord_total_shots'].apply(lambda x: x / total_games) # We assume a game is 1 hour long
    # We assume a game is 1 hour long
    df_rate_shots.rename(columns={'coord_total_shots':'coord_rate_shots'}, inplace=True)
    print(df_rate_shots.head(5))

    return df_rate_shots

def excess_for_team(teamdf:pd.DataFrame, acrossteam:pd.DataFrame):
    """
    This function calculates the excess of shots for a team in a given location
    """
    teamdf=teamdf.copy()
    acrossteam=acrossteam.copy()
    teamdf=shotrate_per_location(teamdf) # Only shots for a team
    acrossteam=shotrate_per_location(acrossteam) # Shots accross all teams

    # Calculate the excess of shots for a team in a given location
    teamdf['excess_shots']=teamdf['coord_rate_shots']-acrossteam['coord_rate_shots']
    return teamdf

df = pd.read_csv("../datasets/Data.csv")
data=df.copy()
# Sample ice rink image (replace with your image)
ice_rink_image_path = '../datasets/left_half.png'  # Replace with your ice rink image


# Load the ice rink image
with open(ice_rink_image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create a function to generate the plot
def generate_plot(selected_team, selected_year):
    team_filtered_data = data[(data['team.name'] == selected_team) &
                            (data['about.dateTime'] == selected_year)]

    accrossteam_filtered_data = data[data['about.dateTime'] == selected_year]

    excess_df=excess_for_team(team_filtered_data, accrossteam_filtered_data)
    excess_df=excess_df.reset_index()

    x, y = np.meshgrid(np.linspace(0,100, 101),
                    np.linspace(-42.5,42.5,81))

    z=griddata((excess_df['coordinates.x'], excess_df['coordinates.y']), excess_df['excess_shots'], (x,y), method='cubic', fill_value=0)
    z=pd.DataFrame(z)*100
    zsmooth=gaussian_filter(z, sigma=6)

    # Load the ice rink image as the background
    img = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    fig, ax = plt.subplots()

    # Display the rink image in the background
    ax.imshow(img, origin='lower', extent=[0, 100, -42.5, 42.5])
    contour = ax.contourf(x, y, zsmooth, cmap='bwr', levels=(20), alpha=0.5, origin='lower', cmin=-1, cmax=1)
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='Différence du taux de tirs par heure', shrink=0.8, aspect=20)
    plt.title(f"Différence du taux de tir par heure pour \n l'équipe {selected_team} en {selected_year} {selected_year+1}")
    plt.xticks(np.arange(0, 100, 10))
    plt.xlabel('Distance des tirs par rapport au but (pieds)')
    plt.ylabel('Largeur de la patinoire (pieds)')

    # Save the plot as a temporary image
    output_buffer = io.BytesIO()
    plt.savefig(output_buffer, format="png")
    output_buffer.seek(0)
    # Exportez le graphique en HTML
    return output_buffer

# Create a Dash application
app = dash.Dash(__name__)
server = app.server
@app.callback(
    Output('shot-map', 'src'),
    [Input('team-dropdown', 'value'), Input('year-dropdown', 'value')]
)
def update_plot(selected_team, selected_year):
    output_buffer = generate_plot(selected_team, selected_year)
    return f"data:image/png;base64,{base64.b64encode(output_buffer.read()).decode('utf-8')}"

# Create dropdown menus for team, season, and year
team_dropdown = dcc.Dropdown(
    id='team-dropdown',
    options=[{'label': team, 'value': team} for team in data['team.name'].unique()]
)

# season_dropdown = dcc.Dropdown(
#     id='season-dropdown',
#     options=[{'label': season, 'value': season} for season in data['Game_type'].unique()],
#     value=data['Game_type'].unique()[0]
# )
year_dropdown = dcc.Dropdown(
    id='year-dropdown',
    options=[{'label': year, 'value': year} for year in data['about.dateTime'].unique()],
    value=data['about.dateTime'].unique()[0]
)


# Create the initial layout with dropdowns and the image
app.layout = html.Div([
    html.H1('NHL Shot Map'),
    team_dropdown,
    year_dropdown,
    html.Img(id='shot-map'),
])


if __name__ == '__main__':
    app.run_server(debug=True)
