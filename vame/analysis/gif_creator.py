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
import tqdm
import umap
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from vame.util.auxiliary import read_config
from vame.util.gif_pose_helper import get_animal_frames
from vame.analysis.tree_hierarchy import graph_to_tree, traverse_tree_cutline, hierarchy_pos
import networkx as nx
from collections import deque
from scipy.spatial.distance import cdist

def get_cluster_echantillon(data_label,time_window, cluster, len_echantillon=5):
    dic_echantillon = {}
    length=len(data_label)
    dic_echantillon[cluster]=[]
    k=0
    while k<length and len(dic_echantillon[cluster])<len_echantillon:
        if data_label[k]==cluster:
            dic_echantillon[cluster].append(k)
            k+=time_window #On ne veut pas de chevauchement
        else :
            k+=1
    return dic_echantillon  

def additione_df(df_a,df_a_2):
    df_a=df_a.reset_index()
    df_a_2=df_a_2.reset_index()
    output=df_a+df_a_2
    output=output.drop("index",axis=1)
    return output

def division(x,n):
    return x/n
            
def create_video(path_to_file, file, embed, clabel, start, length, max_lag, num_points, n_cluster): 
    # set matplotlib colormap
    cmap = matplotlib.cm.gray
    cmap_reversed = matplotlib.cm.get_cmap('gray_r')

    # this here generates every frame for your gif. The gif is lastly created by using ImageJ
    # the embed variable is my umap embedding, which is for the 2D case a 2xn dimensional vector
    fig = plt.figure()
    spec = GridSpec(ncols=2, nrows=1, width_ratios=[6, 3])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1])
    ax2.axis('off')
    ax2.grid(False)
    lag = 0
            #ci dessous pour afficher la colorbar pour le plt avec les clusters mais augmente grandement le temps de traitement
    cmap = cm.get_cmap('Spectral', n_cluster)
    sm = ScalarMappable(norm=colors.Normalize(vmin=0, vmax=n_cluster), cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_ticks(np.arange(0, n_cluster, 1))
    cbar.set_ticklabels(np.arange(0, n_cluster, 1))
    for i in tqdm.tqdm(range(0, length, 1)):
        if i > max_lag:
            lag = i - max_lag
        ax1.cla()
        ax1.grid(False)
        ax1.scatter(embed[:num_points,0], embed[:num_points,1], c=clabel[:num_points], cmap='Spectral', s=1, alpha=0.4)
        ax1.set_aspect('equal', 'datalim')

        ax1.plot(embed[start+lag:start+i,0], embed[start+lag:start+i,1],'.b-',alpha=.6, linewidth=2, markersize=4)
        ax1.plot(embed[start+i,0], embed[start+i,1], 'gx', markersize=4)
        
        fig.savefig(os.path.join(path_to_file,"gif_frames",file+'gif_%d.png') %i) 
                
def create_video_test(path_to_file, file, embed, clabel, start, length, max_lag, num_points, n_cluster,df, time_window): 
    # set matplotlib colormap
    cmap = matplotlib.cm.gray
    cmap_reversed = matplotlib.cm.get_cmap('gray_r')
    sfreq = 10 # sampling frequency [Hz]
    visible = 2000
    #print(path_to_file)
    #images_dir = os.listdir(os.path.join(path_to_file, 'frame_roulage'))
    images = []
    images_abs_path = []

    '''for im in images_dir:
        images_abs_path.append(os.path.abspath(os.path.join(path_to_file, 'frame_roulage', im)))

    images_abs_path = sorted(images_abs_path, key=lambda x: int(x.split("_")[3].split(".")[0]))
    
    print('Sorting images for video...')
    for im in tqdm.tqdm(images_abs_path):
        if im.endswith(".png"):
            images.append(mpimg.imread(im))'''
            
    ################ TREE CREATION ##########################
    transition_matrix = np.load(os.path.join(path_to_file, "community", "transition_matrix_video-1.npy"))
    _, usage = np.unique(clabel, return_counts=True)
    T = graph_to_tree(usage, transition_matrix, n_cluster, merge_sel=1)

    centroids = np.load(os.path.join(path_to_file, "cluster_center_video-1.npy"))
    dq_Distance = deque(np.zeros(visible), visible)
    dq_LongAccCorr = deque(np.zeros(visible), visible)
    dq_TraAccCorr = deque(np.zeros(visible), visible)
    dq_VehSpeed = deque(np.zeros(visible), visible)
    dx = deque(np.zeros(visible), visible)
    interval = np.linspace(0, df.shape[0], num=df.shape[0])
    interval /= sfreq # from samples to seconds
    columns_=["Distance_to_CIPV","LongitudinalAccelCorrected","TransversalAccelCorrected","VehicleSpeed"]
    fig = plt.figure(figsize=(24,16))

    gs = GridSpec(6, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[3, 0])
    ax9 = fig.add_subplot(gs[3, 1])
    ax10 = fig.add_subplot(gs[2, 2])
    ax11 = fig.add_subplot(gs[2, 3])
    ax12 = fig.add_subplot(gs[3, 2])
    ax13 = fig.add_subplot(gs[3, 3])
    ax14 = fig.add_subplot(gs[4:6, 1:3])
    ax15 = fig.add_subplot(gs[4:6, 0])
                           
    ax2.set_xlabel("Time [s]", fontsize=10, labelpad=8)
    ax2.set_ylabel("Distance_to_CIPV [m]", fontsize=10)
    l_distance, = ax2.plot(dx, dq_Distance, color='silver', label=columns_[0])
    ax2.legend(loc="upper right", fontsize=9, fancybox=True, framealpha=0.5)
    
    ax3.set_xlabel("Time [s]", fontsize=10, labelpad=8)
    ax3.set_ylabel("LongitudinalAccelCorrected [g]", fontsize=10)
    l_longacc, = ax3.plot(dx, dq_LongAccCorr, color='silver', label=columns_[1])
    ax3.legend(loc="upper right", fontsize=9, fancybox=True, framealpha=0.5)
    
    ax4.set_ylabel("TransversalAccelCorrected [g]", fontsize=10)
    l_transacc, = ax4.plot(dx, dq_TraAccCorr, color='silver', label=columns_[2])
    ax4.legend(loc="upper right", fontsize=9, fancybox=True, framealpha=0.5)
    
    ax5.set_ylabel("VehicleSpeed [km/h]", fontsize=10)
    l_vehspeed, = ax5.plot(dx, dq_TraAccCorr, color='silver', label=columns_[3])
    ax5.legend(loc="upper right", fontsize=9, fancybox=True, framealpha=0.5)   
    
    lag = 0
    step = 1
    color = 0
    start_ts = 0
    cmap = cm.get_cmap('Spectral', n_cluster)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=n_cluster), cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_ticks(np.arange(0, n_cluster, 1))
    cbar.set_ticklabels(np.arange(0, n_cluster, 1))
    cluster_buffer = -1
    for i in tqdm.tqdm(range(0, length, step)):
        if i > max_lag:
            lag = i - max_lag
        
        if cluster_buffer == -1:
            cluster_buffer = clabel[start+i]
        
        if clabel[start+(i-step)] != clabel[start+i]:
            cluster_buffer = clabel[start+(i-step)]
        
############# PLOT FOR TSNE/UMAP ##################
        ax1.cla()
        ax1.grid(False)
        scatter = ax1.scatter(embed[:num_points,0], embed[:num_points,1], c=clabel[:num_points], cmap='Spectral', s=1, alpha=0.4, label=f"Cluster n°{clabel[start+i]}")
        ax1.legend(loc="upper right", fontsize=9, fancybox=True, framealpha=0.5)
        #if i >= 1:
        centroid_distance = round(cdist([centroids[clabel[start+i]]], [centroids[clabel[start+(i-step)]]])[0][0], 3)
        ax1.set_title(f"Current cluster : {clabel[start+i]}, Previous cluster : {clabel[start+(i-step)] if clabel[start+(i-step)] != clabel[start+i] else 'Same'}, Mean distance between cluster:{centroid_distance}", loc = "center")
        ax1.set_aspect('equal', 'datalim')
        ax1.plot(embed[start+i,0], embed[start+i,1], 'gx', markersize=10)
        ax1.plot(embed[start+lag:start+i,0], embed[start+lag:start+i,1],'.b-',alpha=.6, linewidth=2, markersize=4)

        
############# PLOTS FOR REAL TIME TIME SERIES REPRESENTATION ##################
  
        dx.extend(interval[start+start_ts:start+start_ts+visible])
        
        dq_Distance.extend(df[columns_[0]].iloc[start+start_ts:start+start_ts+visible])
        l_distance.set_ydata(dq_Distance)  
        l_distance.set_xdata(dx)
        m_dq_Distance = np.mean(dq_Distance)
        ax2.set_ylim(-30+m_dq_Distance, 80+m_dq_Distance)
        ax2.set_xlim(interval[start+start_ts], interval[start+start_ts+visible]) 
        
        dq_LongAccCorr.extend(df[columns_[1]].iloc[start+start_ts:start+start_ts+visible])
        l_longacc.set_ydata(dq_LongAccCorr)  
        l_longacc.set_xdata(dx)
        m_dq_LongAccCorr = np.mean(dq_LongAccCorr)
        ax3.set_ylim(-0.3+m_dq_LongAccCorr, 0.3+m_dq_LongAccCorr)
        ax3.set_yscale('linear')
        ax3.set_xlim(interval[start+start_ts], interval[start+start_ts+visible])
        
        dq_TraAccCorr.extend(df[columns_[2]].iloc[start+start_ts:start+start_ts+visible])
        l_transacc.set_ydata(dq_TraAccCorr)  
        l_transacc.set_xdata(dx)
        m_dq_TraAccCorr = np.mean(dq_TraAccCorr)
        ax4.set_yscale('linear')
        ax4.set_ylim(-0.3+m_dq_TraAccCorr, 0.3+m_dq_TraAccCorr)
        ax4.set_xlim(interval[start+start_ts], interval[start+start_ts+visible])
        
        dq_VehSpeed.extend(df[columns_[3]].iloc[start+start_ts:start+start_ts+visible])
        l_vehspeed.set_ydata(dq_VehSpeed)  
        l_vehspeed.set_xdata(dx)
        m_dq_VehSpeed = np.mean(dq_VehSpeed)
        ax5.set_ylim(-120+m_dq_VehSpeed, 50+m_dq_VehSpeed)
        ax5.set_xlim(interval[start+start_ts], interval[start+start_ts+visible])        
         
        start_ts+=10
        
############# PLOTS FOR AVERAGE ON CURRENT CLUSTER ##################
        cluster_echantillon = get_cluster_echantillon(clabel, time_window, clabel[start+i], len_echantillon = 5)
        value=cluster_echantillon[clabel[start+i]]
        ax_num = 0
        list_axes = [ax6, ax7, ax8, ax9]
        for column in columns_:
            y_min=df[column].min()
            y_max=df[column].max()
            temp_add=df[value[0]:value[0]+time_window][column]
            n_echantillons=len(value)
            if n_echantillons==0:
                continue
            for j in range(1,n_echantillons):
                temp=df[value[j]:value[j]+time_window][column]
                temp_add=additione_df(temp_add,temp)
            temp=temp_add.apply(division,n=n_echantillons)
            x=np.arange(0,len(temp)/10,0.1)
            list_axes[ax_num].cla()
            list_axes[ax_num].plot(x, temp,label=column) #On ne labellise que la première pour éviter la redondance
            list_axes[ax_num].set_ylim(y_min,y_max)
            list_axes[ax_num].set_title(f"{column} average on current cluster : {clabel[start+i]}", loc='center')
            #list_axes[ax_num].set_ylabel(column, fontsize=10)
            ax_num += 1
            
############# PLOTS FOR AVERAGE ON PREVIOUS CLUSTER ##################
        list_axes = [ax10, ax11, ax12, ax13]
        cluster_echantillon = get_cluster_echantillon(clabel, time_window, cluster_buffer, len_echantillon = 5)  
        value=cluster_echantillon[cluster_buffer]
        ax_num = 0
        for column in columns_:
            y_min=df[column].min()
            y_max=df[column].max()
            temp_add=df[value[0]:value[0]+time_window][column]
            n_echantillons=len(value)
            if n_echantillons==0:
                continue
            for j in range(1,n_echantillons):
                temp=df[value[j]:value[j]+time_window][column]
                temp_add=additione_df(temp_add,temp)
            temp=temp_add.apply(division,n=n_echantillons)
            x=np.arange(0,len(temp)/10,0.1)
            list_axes[ax_num].cla()
            list_axes[ax_num].plot(x, temp,label=column) #On ne labellise que la première pour éviter la redondance
            list_axes[ax_num].set_ylim(y_min,y_max)
            list_axes[ax_num].set_title(f"{column} average on previous cluster : {cluster_buffer}", loc='center')
            ax_num += 1


                
        ax6.set_ylabel("Distance_to_CIPV [m]", fontsize=10)
        ax7.set_ylabel("LongitudinalAccelCorrected [g]", fontsize=10)
        ax8.set_ylabel("TransversalAccelCorrected [g]", fontsize=10)
        ax8.set_xlabel("Time [s]", fontsize=10, labelpad=8)
        ax9.set_ylabel("VehicleSpeed [km/h]", fontsize=10)
        ax9.set_xlabel("Time [s]", fontsize=10, labelpad=8)
        ax10.set_ylabel("Distance_to_CIPV [m]", fontsize=10)
        ax11.set_ylabel("LongitudinalAccelCorrected [g]", fontsize=10)
        ax12.set_ylabel("TransversalAccelCorrected [g]", fontsize=10)
        ax12.set_xlabel("Time [s]", fontsize=10, labelpad=8)
        ax13.set_ylabel("VehicleSpeed [km/h]", fontsize=10)
        ax13.set_xlabel("Time [s]", fontsize=10, labelpad=8)
        
################# DRAWING TREE ########################
        ax14.cla()
        cmap = scatter.get_cmap()
        colors = cmap(np.arange(cmap.N))
        hex_colors = [mcolors.to_hex(color) for color in colors]
        pos = hierarchy_pos(T, 'Root', width=.5, vert_gap=0.1, vert_loc=0, xcenter=50)
        node_colors = ['red' if node == clabel[start+i] else 'blue' if node == cluster_buffer else 'gray' for node in T.nodes()]
        nx.draw_networkx(T, pos, node_color=node_colors, ax=ax14, label='Blue is previous cluster, Red is current cluster')
        shortest_path = nx.shortest_path(T, source=clabel[start+i], target=cluster_buffer)
        ax14.set_title(f'Distance in the tree between current cluster and previous cluster is : {len(shortest_path) - 1}')

############ Adding images ##############
        ax15.cla()
        ax15.axis('off')
        #if i < len(images):
        #    ax15.imshow(images[i])
        ax15.set_title('Vidéo roulage')
        #saving each fig at each step of the loop to generate frames that wil be used to create video later    
        fig.tight_layout()
        fig.savefig(os.path.join(path_to_file,"gif_frames",file+'gif_%d.png') %(start+i)) 
        


def gif(config, df, pose_ref_index, subtract_background=True, start=None, length=500, 
        max_lag=30, label='community', file_format='.mp4', crop_size=(300,300)):
    
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
        

    for file in files:
        path_to_file=os.path.join(cfg['project_path'],"results",file,model_name,'kmeans-'+str(n_cluster),"")
        if not os.path.exists(os.path.join(path_to_file,"gif_frames")):
            os.mkdir(os.path.join(path_to_file,"gif_frames"))
        
        embed = np.load(os.path.join(path_to_file,"community","umap_embedding_"+file+'.npy'))
        
        try:
            embed = np.load(os.path.join(path_to_file,"","community","","umap_embedding_"+file+".npy"))
            num_points = cfg['num_points']
            if num_points > embed.shape[0]:
                num_points = embed.shape[0]
        except:
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
        
        if label == "motif":
            umap_label = np.load(os.path.join(path_to_file,str(n_cluster)+"_km_label_"+file+'.npy'))
        elif label == "community":
            umap_label = np.load(os.path.join(path_to_file,"community","community_label_"+file+'.npy'))
        elif label == None:
            umap_label = None
        
        if start == None:
            start = np.random.choice(embed[:num_points].shape[0]-length)
        else:
            start = start
        
        #frames = get_animal_frames(cfg, file, pose_ref_index, start, length, subtract_background, file_format, crop_size)
        time_window = cfg['time_window']
        create_video_test(path_to_file, file, embed, umap_label, start, length, max_lag, num_points, n_cluster, df, time_window)
                   
        

























