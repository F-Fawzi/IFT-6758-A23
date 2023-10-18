# IFT-6758-A23  
Projet science des données de la LNH  « play-by-play » : De manière générale, le schéma de cette étape est d'utiliser l'API de statistiques de la LNH pour récupérer à la fois des données agrégées (statistiques des joueurs pour une saison donnée) et des données « play-by-play » pour une période de temps spécifique. On a commencé par créer des visualisations simples à partir des données agrégées qui ne nécessitent pas beaucoup de prétraitement. Ensuite, on a créer des visualisations interactives à partir des données play-by-play. Il y aura un petit nombre de questions qualitatives simples auxquelles répondre tout au long des tâches qui seront liées aux tâches décrites. Enfin, nous avons présenter votre travail sous la forme d'une simple page web statique, créée à l'aide de Jekyll.  

# Dossiers
| Noms fichiers | Description |
| ------------- | ------------- |
| AcquisitionDonnées.py | Une classe pour télécharger les données play-by-play de la LNH pour la saison régulière et les séries éliminatoires.(.py) |
| nettoyage.py |Une fonction pour convertir tous les événements de chaque jeu en dataframe Pandas.(.py) |
| Visualisations_Simples.ipynb | Implémentation d'un ipywidget qui permet de parcourir tous les événements, pour chaque match d'une saison donnée, avec la possibilité de changer entre la saison régulière et les séries éliminatoires.(.ipynb) |
| VisualisationsAvancées.ipynb |Plans de tir pour une équipe de la LNH donnée, pour une année(.ipynb) |


## Export visualisation HTML  
Dash est un package Python Dashboard open source de plot.ly. Il est assez facile à utiliser et contient de nombreux composants permettant de créer des graphiques et des diagrammes magnifiques et informatifs. Afin de visualiser notre visualisation avancées nous avons utiliser Gunicorn pour le déploiement dans un environnement de production.   Vous pouvez voir la structure de dossier a respecter :  
* src : dossier qui contient l'application app.
* datasets : les données nécessaires pour l'exécution de l'app.  
Puis, allez sur votre Render, sélectionnez un webservice donnez un nom à votre service, cela fera partie de votre URL publique et vérifiez les paramètres, spécialement la commande run.  







# Références : 
https://www.youtube.com/watch?v=OBGaCULCZzg&ab_channel=RajKapadia
https://plotly.com/python/interactive-html-export/
