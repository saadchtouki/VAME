#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 1.0-alpha Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""


import os
#import umap
import umap.umap_ as umap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from vame.util.auxiliary import read_config
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def get_label_indexes(label,n_cluster):
    dic={}
    labels_found=[]
    for i in range(len(label)):
        label_=label[i]
        if label_ not in labels_found:
            dic[label_]=i
            labels_found.append(label_)
            if len(labels_found)==n_cluster:
                break
    return dic

def umap_vis(file, embed, num_points):        
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1], s=2, alpha=.5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    

def umap_label_vis(file, embed, label, n_cluster, num_points):
    fig = plt.figure(1)
    #plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=label[:num_points], cmap='Spectral', s=2, alpha=.7)
    scatter=plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=label[:num_points], cmap='Spectral', s=2, alpha=.7)
    
    #plt.colorbar(boundaries=np.arange(n_cluster+1)-0.5).set_ticks(np.arange(n_cluster))
    cb=plt.colorbar(boundaries=np.arange(n_cluster+1)-0.5)
    cb.set_ticks(np.arange(n_cluster))
    
    dic=get_label_indexes(label,n_cluster)
    cluster_colors=[]
    for i in range(n_cluster):
        point_color = scatter.to_rgba(label[dic[i]])
        cluster_colors.append(point_color)
    
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    return (cluster_colors,fig)

def umap_vis_comm(file, embed, community_label, num_points):
    num = np.unique(community_label).shape[0]
    fig = plt.figure(1)
    plt.scatter(embed[:num_points,0], embed[:num_points,1],  c=community_label[:num_points], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(boundaries=np.arange(num+1)-0.5).set_ticks(np.arange(num))
    plt.gca().set_aspect('equal', 'datalim')
    plt.grid(False)
    

def visualization(config, label=None):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = cfg['n_cluster']
    
    files = []
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"
                     "If you only want to use a specific dataset type filename: \n"
                     "yes/no/filename ")
    else:
        all_flag = 'yes'

    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)

    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        files.append(all_flag)

    for idx, file in enumerate(files):
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",'kmeans-'+str(n_cluster))
        
        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
            num_points = cfg['num_points']
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except:
            if not os.path.exists(os.path.join(path_to_file,"community")):
                os.mkdir(os.path.join(path_to_file,"community"))
            print("Compute embedding for file %s" %file)
            reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                    random_state=cfg['random_state']) 
            
            latent_vector = np.load(os.path.join(path_to_file,"",'latent_vector_'+file+'.npy'))
            
            num_points = cfg['num_points']
            if num_points > latent_vector.shape[0]:
                num_points = latent_vector.shape[0]
            print("Embedding %d data points.." %num_points)
            
            embed = reducer.fit_transform(latent_vector[:num_points,:])
            np.save(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'), embed)
        
        print("Visualizing %d data points.. " %num_points)
        if label == None:                    
            umap_vis(file, embed, num_points)
            
        if label == 'motif':
            motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+file+'.npy'))
            (cluster_colors,fig)=umap_label_vis(file, embed, motif_label, n_cluster, num_points) 
            if not os.path.exists(os.path.join(cfg['project_path'],'results','video-1','VAME',"kmeans-"+str(n_cluster),'Latent_visualization')): #Ajouté
                os.mkdir(os.path.join(cfg['project_path'],'results','video-1','VAME',"kmeans-"+str(n_cluster),'Latent_visualization')) #Ajouté
            np.save(os.path.join(cfg['project_path'],'results','video-1','VAME',"kmeans-"+str(n_cluster),'Latent_visualization','cluster_colors.npy'),cluster_colors) #Ajouté
            fig.savefig(os.path.join(cfg['project_path'],'results','video-1','VAME',"kmeans-"+str(n_cluster),'Latent_visualization','Latent_figure.png')) #Ajouté

        if label == "community":
            community_label = np.load(os.path.join(path_to_file,"","community","","community_label_"+file+".npy"))
            umap_vis_comm(file, embed, community_label, num_points) 

            

def silhouette_N_Umap_vis(file, embed, data_label, n_clusters, num_points, len_df, sample_silhouette_values, silhouette_avg):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(23, 12)
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette    
    # plots of individual clusters, to demarcate them clearly.    
    ax1.set_ylim([0, len_df + (n_clusters + 1) * 10])
    y_lower = 20    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to        
        # cluster i, and sort them        
        ith_cluster_silhouette_values = sample_silhouette_values[data_label == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle        
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot        
        y_lower = y_upper + 20    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical line for average silhouette score of all the values    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks    
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    mappable = plt.cm.ScalarMappable(cmap='Spectral')
    mappable.set_array(data_label[:num_points])
    mappable.set_clim(0, n_clusters-1)
    ax2.scatter(embed[:num_points,0], embed[:num_points,1],  c=data_label[:num_points], cmap='Spectral', s=2, alpha=.7)
    plt.colorbar(mappable, ax=ax2)
    plt.gca().set_aspect('equal', 'datalim')
    ax2.set_title("The visualization of the clustered data.")
    plt.grid(False)
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()          
          

def visualization_for_silhouette(config, label, len_df, n_clusters, sample_silhouette_values, silhouette_avg):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg['model_name']
    n_cluster = n_clusters    
    files = []
    
    if cfg['all_data'] == 'No':
        all_flag = input("Do you want to write motif videos for your entire dataset? \n"                     
                         "If you only want to use a specific dataset type filename: \n"                     
                         "yes/no/filename ")
    else:
        all_flag = 'yes' 
        
    if all_flag == 'yes' or all_flag == 'Yes':
        for file in cfg['video_sets']:
            files.append(file)
                
    elif all_flag == 'no' or all_flag == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to quantify " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue    
    else:
        files.append(all_flag)
        
    for idx, file in enumerate(files):
        path_to_file=os.path.join(cfg['project_path'],"results",file,"",model_name,"",'kmeans-'+str(n_cluster))
        
        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
            num_points = cfg['num_points']
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except:
            if not os.path.exists(os.path.join(path_to_file,"community")):
                os.mkdir(os.path.join(path_to_file,"community"))
            print("Compute embedding for file %s" %file)
            reducer = umap.UMAP(n_components=2, min_dist=cfg['min_dist'], n_neighbors=cfg['n_neighbors'], 
                    random_state=cfg['random_state']) 
            latent_vector = np.load(os.path.join(path_to_file,"",'latent_vector_'+file+'.npy'))
            num_points = cfg['num_points']
            if num_points > latent_vector.shape[0]:
                num_points = latent_vector.shape[0]
            print("Embedding %d data points.." %num_points)
            embed = reducer.fit_transform(latent_vector[:num_points,:])
            np.save(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'), embed)
        print("Visualizing %d data points.. " %num_points)
        if label == None:                    
            umap_vis(file, embed, num_points)
        if label == 'motif':
            motif_label = np.load(os.path.join(path_to_file,"",str(n_cluster)+'_km_label_'+file+'.npy'))
            #umap_label_vis(file, embed, motif_label, n_cluster, num_points)            
            silhouette_N_Umap_vis(file, embed, motif_label, n_cluster, num_points, len_df, sample_silhouette_values, silhouette_avg)
        if label == "community":
            community_label = np.load(os.path.join(path_to_file,"","community","","community_label_"+file+".npy"))
            umap_vis_comm(file, embed, community_label, num_points)













