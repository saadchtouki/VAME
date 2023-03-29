
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:03:41 2023

@author: saadchtouki
"""

import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import base64
import json
import os
from pathlib import Path
import vame
from vame.util.auxiliary import read_config
import numpy as np


print("")
print("Veuiller vérifier que le bon nombre de clusters est fixé dans congig.yaml")
print("")
config=input("Entrer le lien du fichier config que vous souhaitez : ")
config_file = Path(config).resolve()

while not config_file.is_file():
    print("Le lien de fichier entré n'est pas valide. Veuillez réessayer.")
    config=input("Entrer le lien du fichier config que vous souhaitez : ")
    config_file = Path(config).resolve()

# Create dash app
app = dash.Dash(__name__)

#edges=[('Root', 'h_0_28'), ('Root', 'h_1_28'), ('h_0_28', 6), 
#hierarchy={'Root': (50, 0), 'h_0_28': (49.875, -0.1), 6: (49.8125, -0.2)


cfg = read_config(config_file) 
project_path=cfg['project_path']
n_cluster=cfg['n_cluster']

hierarchy_edges_name='hierarchy_edges'
with open(os.path.join(project_path,'results','video-1','VAME','kmeans-'+str(n_cluster),'Hierarchy_&_Egdes',hierarchy_edges_name)) as f:
    hierarchy_edges = json.load(f)
    
hierarchy=hierarchy_edges[0]
edges=hierarchy_edges[1]
#print(hierarchy)
#print(edges)

xe_=[]
ye_=[]
for edge in edges :
    (a,b)=edge
    #print(a,b)
    xe_.append(hierarchy[str(a)][0])
    ye_.append(hierarchy[str(a)][1])
    xe_.append(hierarchy[str(b)][0])
    ye_.append(hierarchy[str(b)][1])
    xe_.append(None)
    ye_.append(None)


x_=[]
y_=[]
labels=[]
for e in hierarchy:
    labels.append(e)
    (a,b)=hierarchy[e]
    x_.append(a)
    y_.append(b)

cluster_colors=np.load(os.path.join(project_path,'results','video-1','VAME','kmeans-'+str(n_cluster),'Latent_visualization','cluster_colors.npy')) #couleurs par cluster

colors=[]
for label in labels:
    if label.isdigit():
        colors.append('rgba('+str(cluster_colors[int(label)][0])+', '+str(cluster_colors[int(label)][1])+', '+str(cluster_colors[int(label)][2])+', '+str(cluster_colors[int(label)][3])+')')
    else:
        colors.append('rgba(0.6, 0.6, 0.6, 1)')
        
# Set dog and cat images
dogImage = "https://www.iconexperience.com/_img/v_collection_png/256x256/shadow/dog.png" #Only useful for size purposes
catImage = "https://d2ph5fj80uercy.cloudfront.net/06/cat3602.jpg" #Only size useful for size purposes

path_tsne=os.path.join(project_path,'results','video-1','VAME','kmeans-'+str(n_cluster),'Latent_visualization','Latent_figure.png')

pics=[]

for cluster in range (n_cluster):
    pics.append(os.path.join(project_path,'results','video-1','VAME','kmeans-'+str(n_cluster),'Plots','cluster_'+str(cluster)+'.png'))
pics.append(os.path.join("Images",'White_Image.png'))
#dogImage = "Results/Result_1.png"
#catImage = "Results/Result_2.png"

images_=[]
for node in hierarchy:
    if node.isdigit():
        images_.append(pics[int(node)])
    else:
        images_.append(pics[n_cluster]) #Adding a WhiteImage for the nodes which aren't clusters


# Generate dataframe
df = pd.DataFrame(
   dict(
      x=x_,
      y=y_,
      images=images_,
      customdata=['{}<br><img src="{}">'.format(e, dogImage if i%2==0 else catImage) for i, e in enumerate(hierarchy)]
   )
)

# Create scatter plot with x and y coordinates
#fig = px.scatter(df, x="x", y="y",custom_data=["images"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=xe_,
                   y=ye_,
                   name="",
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'))
fig.add_trace(go.Scatter(x=x_,
                  y=y_,
                  mode='markers',
                  name="",#'bla',
                  marker=dict(symbol='circle-dot',
                                size=18,
                                color=colors,    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8,
                  customdata=df["images"]
                  ))

fig.update_layout(
    xaxis=dict(
        tickvals=[],
        ticktext=[]
    ),
    yaxis=dict(
        tickvals=[],
        ticktext=[]
    ),
    showlegend=False
)

path_banner=os.path.join("Images","banniere.jpg")

with open(path_banner, 'rb') as f:
    banner_binary = f.read()
    encoded_banner = base64.b64encode(banner_binary).decode('ascii')
banner_image='data:image/png;base64,{}'.format(encoded_banner)

with open(path_tsne, 'rb') as f:
    tsne_binary = f.read()
    encoded_tsne = base64.b64encode(tsne_binary).decode('ascii')
still_image='data:image/png;base64,{}'.format(encoded_tsne)

# Update layout and update traces
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=20)

# Create app layout to show dash graph
app.layout = html.Div(
   [
       html.Div([
          html.Img(src=banner_image, style={'width': '100%', 'display': 'inline-block'})])
      ,dcc.Graph(
         id="graph_interaction",
         figure=fig,
      ),
      html.Div([
          html.Img(id='current-image', style={'width': '33%', 'display': 'inline-block'}),
        html.Img(id='previous-image', style={'width': '33%', 'display': 'inline-block'}),
                html.Img(src=still_image, style={'width': '33%', 'display': 'inline-block'})
                
    ])])
   
#Définition de still image: qui va être le TSNE




previous_image = None

@app.callback(
   [Output('current-image', 'src'),
     Output('previous-image', 'src')],
   Input('graph_interaction', 'hoverData'))
def open_local_image(hoverData):
    global previous_image#, still_image
    if hoverData:
        # Récupérer le chemin d'accès local de l'image
        image_path = hoverData["points"][0]["customdata"]
        # Ouvrir le fichier image en mode binaire et le lire 
        with open(image_path, 'rb') as f:
            image_binary = f.read()
            # Convertir l'image en base64
            encoded_image = base64.b64encode(image_binary).decode('ascii')
        # Retourner la source de l'image encodée en base64
        current_image = 'data:image/png;base64,{}'.format(encoded_image)
        
        # Store current image as previous image
        previous_value_image= previous_image
        
        if image_path != os.path.join("Images,'White_Image.png'):
            previous_image = current_image
        
        # Return current and previous images
        return [current_image, previous_value_image]
    #else:
        #raise PreventUpdate
    
   
   #   

if __name__ == '__main__':
   app.run_server(debug=True)
