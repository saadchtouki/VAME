from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vame
import numpy
import ruamel
import os
from pathlib import Path
import scipy.signal
from scipy.stats import iqr
import matplotlib.pyplot as plt
from datetime import datetime as dt
from vame.util.auxiliary import read_config
import torch
import math
from math import floor
import random
from vame.analysis.tree_hierarchy import graph_to_tree, draw_tree, traverse_tree_cutline, hierarchy_pos
import json
import yaml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


path_parquet=#Enter the path to the parquet file here

def scale(data, standard=['Distance_to_CIPV','LongitudinalAccelCorrected','TransversalAccelCorrected','YawRateCorrected'], minmax=['VehicleSpeed']):
    df=data.copy()
    MinMax_scaler = MinMaxScaler(feature_range=(-1,1))
    Standard_scaler = StandardScaler()
    for col in standard :
        df[col]= Standard_scaler.fit_transform(np.array(df[col]).reshape(-1,1))
    for col in minmax:
        df[col]= MinMax_scaler.fit_transform(np.array(df[col]).reshape(-1,1))
    return df

def scaleOnlyMinMax(data, minmax=['VehicleSpeed','Distance_to_CIPV','LongitudinalAccelCorrected','TransversalAccelCorrected','YawRateCorrected']):
    df=data.copy()
    MinMax_scaler = MinMaxScaler(feature_range=(-3,3))
    for col in data.columns:
        if col in minmax:
            df[col]= MinMax_scaler.fit_transform(np.array(df[col]).reshape(-1,1))
    return df

def open_scale_clean_parquet(num_parquet, columns, scaling=True):
    path=os.path.join(path_parquet,"input_full_hdd_"+num_parquet+".parquet")
    a=pd.read_parquet(path)
    data=a[columns]
    data.replace("",np.nan,inplace=True)
    data=data.interpolate(method='nearest', limit=10)#ImputationValeursManquantes
    #data=scale(data) #scaling
    if scaling:
        data = scaleOnlyMinMax(data)
    return data

def compute_deviation_score_speed_deviation_and_length(data,trip): #Return (deviation_score,speed_deviation, length)
    deviation_score=1
    for i in data[data['trip'].isin([trip])].std():
        deviation_score=deviation_score*i
    speed_deviation=data[data['trip'].isin([trip])]['VehicleSpeed'].std()
    length= len(data[data['trip'].isin([trip])])
    return (deviation_score, speed_deviation, length)

def show_parquet_stats(num_parquet,columns): #Columns must include 'trip'  #Helps to pick relevant trips
    path=os.path.join(path_parquet,"input_full_hdd_"+num_parquet+".parquet")
    a=pd.read_parquet(path)
    data=a[columns]
    data.replace("",np.nan,inplace=True)
    data=data.interpolate(method='nearest', limit=10)#ImputationValeursManquantes
    output_columns=['trip','Deviation Score','Speed Deviation', 'length']
    output=pd.DataFrame(columns=output_columns)
    for trip in pd.unique(data['trip']):
        (deviation_score, speed_deviation, length)= compute_deviation_score_speed_deviation_and_length(data, trip)
        df_=pd.DataFrame([[trip, deviation_score, speed_deviation, length]], columns=output_columns)
        output=output.append(df_, ignore_index=True)
        output=output.sort_values(by='Deviation Score',ascending=False)
    return output

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpol(arr):
    y = np.transpose(arr)
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    arr = np.transpose(y)
    return arr

def Create_Trainset_modified(config):
    check_parameter=False
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    legacy = cfg['legacy']
    fixed = cfg['egocentric_data']

    if not os.path.exists(os.path.join(cfg['project_path'],'data','train',"")):
        os.mkdir(os.path.join(cfg['project_path'],'data','train',""))

    files = []
    if cfg['all_data'] == 'No':
        for file in cfg['video_sets']:
            use_file = input("Do you want to train on " + file + "? yes/no: ")
            if use_file == 'yes':
                files.append(file)
            if use_file == 'no':
                continue
    else:
        for file in cfg['video_sets']:
            files.append(file)


    #traindata_fixed(cfg, files, cfg['test_fraction'], cfg['num_features'], cfg['savgol_filter'], check_parameter)

    if check_parameter == False:
        print("A training and test set has been created. Next step: vame.train_model()")
    savgol_filter=cfg['savgol_filter']
    testfraction=cfg['test_fraction']

    X_train = []
    pos = []
    pos_temp = 0
    pos.append(0)


    file='video-1'
    print("z-scoring of file %s" %file)
    path_to_file = os.path.join(cfg['project_path'],"data", file, file+'-PE-seq.npy')
    data = np.load(path_to_file, allow_pickle=True) #Added : allow_pickle=true
    #X_mean = np.nanmean(data,axis=None)  #Ligne supprimée
    #X_std = np.nanstd(data, axis=None)   #Ligne supprimée
    #X_z = (data.T - X_mean) / X_std      #Ligne supprimée
    X_z=data.T


    #Interpolation réussie
    for i in range(X_z.shape[0]):
        X_z[i,:] = interpol(X_z[i,:])
    X_len = len(data.T)
    pos_temp += X_len
    pos.append(pos_temp)
    X_train=[]
    #X_train.append(X_z) ####Inutile car ici entrainement sur un seul fichier

    X = X_z
    #X_2 = np.concatenate(X_train,axis=0).T   ##Inutile 
    X_med = scipy.signal.savgol_filter(X, cfg['savgol_length'], cfg['savgol_order']) 

    num_frames = len(X_med.T)
    test = int(num_frames*testfraction)

    z_test =X_med[:,:test]
    z_train = X_med[:,test:]

    if check_parameter == True:
        plot_check_parameter(cfg, iqr_val, num_frames, X_true, X_med)

    
    #save numpy arrays the the test/train info:
    np.save(os.path.join(cfg['project_path'],"data", "train",'train_seq.npy'), z_train)
    np.save(os.path.join(cfg['project_path'],"data", "train", 'test_seq.npy'), z_test)

    for i, file in enumerate(files):
        np.save(os.path.join(cfg['project_path'],"data", file, file+'-PE-seq-clean.npy'), X_med[:,pos[i]:pos[i+1]])

    print('Lenght of train data: %d' %len(z_train.T))
    print('Lenght of test data: %d' %len(z_test.T))

def add_zero_lines(Path): #Adding zero lines to match the training size
    seq=np.load(Path,allow_pickle=True)
    (a,b)=seq.shape
    if a>=12:
        return seq
    else:
        nb_of_zero_lines=12-a
        length=len(seq[0])
        row=np.zeros(length)
        for i in range(nb_of_zero_lines):
            seq=np.concatenate((seq,[row]), axis=0)
    return seq

def Vame_parquet(num_parquet, columns, trips, name=""):    
    working_directory = os.path.join('VAME','VAME')
    if not (name==""):
        project='VAME-'+num_parquet+'-'+name
    else:
        project='VAME-'+num_parquet
    videos = [os.path.join('VAME','VAME','video-1.csv')]
    
    # Project Initialization
    config = vame.init_new_project(project=project, videos=videos, working_directory=working_directory, videotype='.mp4')
    if config==None:
        print("")
        print("You can't create two projects with the same name during the same day !")
        return
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    
    dataOnlyMinMax = open_scale_clean_parquet(num_parquet,columns)
    data1=dataOnlyMinMax[dataOnlyMinMax['trip'].isin(trips)]
    data1=data1.drop("trip",axis=1)
    
    date = dt.today() #Looking for the date
    month = date.strftime("%B")
    day = date.day
    year = date.year
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    project_name = '{pn}-{date}'.format(pn=project, date=d+'-'+str(year))
    
    if not os.path.exists(os.path.join(cfg['project_path'],'data','video-1',"")):
        os.mkdir(os.path.join(cfg['project_path'],'data','video-1',""))
    np.save(os.path.join('VAME', 'VAME', project_name,'data', 'video-1','video-1-PE-seq.npy'),data1)
    
    
    #Trainset creation
    Create_Trainset_modified(config)
    
    #Adding zero lines
    new_test=add_zero_lines(os.path.join('VAME', 'VAME', project_name,'data', 'train','test_seq.npy'))
    new_train=add_zero_lines(os.path.join('VAME', 'VAME', project_name,'data', 'train','train_seq.npy'))
    seq_clean=np.concatenate((new_test,new_train),axis=1)
    np.save(os.path.join('VAME', 'VAME', project_name,'data', 'train', 'train_seq.npy'),new_train)   #Train saved
    np.save(os.path.join('VAME', 'VAME', project_name,'data', 'train', 'test_seq.npy'),new_test) #Test saved
    np.save(os.path.join('VAME', 'VAME', project_name,'data', 'video-1', 'video-1-PE-seq-clean.npy'),seq_clean) #Clean sequence saved, ready to be trained.
    print("")
    print("")
    print("config = '"+config+"'")
    
def get_cluster_echantillon(data_label,config, len_echantillon=5):
    dic_echantillon = {}
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    length=len(data_label)
    time_window = cfg['time_window']
    for cluster in range(cfg["n_cluster"]):
        dic_echantillon[cluster]=[]
        k=0
        while k<length and len(dic_echantillon[cluster])<len_echantillon:
            if data_label[k]==cluster:
                dic_echantillon[cluster].append(k)
                k+=time_window #On ne veut pas de chevauchement
            else :
                k+=1
    return dic_echantillon  

def plot_cluster_one_by_one(cluster_echantillon, df, columns_, n_clusters, colors):  
    for key in range (len(cluster_echantillon)):
        value=cluster_echantillon[key]
        print(f"------------------------------------ CLUSTER {key} ---------------------------------------")
        for column in columns_:
            for j in range(0,len(value)):
                temp=df[value[j]:value[j]+30]
                x=np.arange(len(temp))
                plt.plot(x, temp[column],color=colors[column])
            plt.legend([column])
            plt.show()    

            
def heaviside(x):
    if x<=0:
        return(0)
    else:
        return(1)

def plot_par_cluster(cluster_echantillon, df, columns_, config, mean=True, show=True, save=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    time_window = cfg['time_window']
    n_clusters=cfg['n_cluster']
    colors = ['red', 'orange' , 'navy', 'teal', 'purple', 'lime', 'aqua', 'blue', 'maroon', 'gray', 'silver', 'fuchsia', 'green', 'yellow', 'olive']
    for key in range (len(cluster_echantillon)):
        value=cluster_echantillon[key]
        if show:
            print("-----------------------------------------------------------------------------")
        total=len(columns_)
        dim=floor(math.sqrt(total))
        rows=total//dim+heaviside(total%dim) #Concerne l'affichage final des graphes (utilisation de la fonction subplot)
        cols=dim
        fig, ax = plt.subplots(rows, cols)
        fig.suptitle(f"Cluster {key}")
        i,t=0,-1 #Ligne et colonne d'affichage (utilisation de la fonction subplot)
        color=0
        for column in columns_:
            y_min=df[column].min()
            y_max=df[column].max()
            if t<cols-1:
                t+=1 #On avance à droite
            else :
                t=0
                i+=1 #On descend d'une ligne
            if mean:
                temp_add=df[value[0]:value[0]+time_window][column]
                n_echantillons=len(value)
                if n_echantillons==0:
                    continue
                for j in range(1,n_echantillons):
                    temp=df[value[j]:value[j]+time_window][column]
                    temp_add=additione_df(temp_add,temp)
                temp=temp_add.apply(division,n=n_echantillons)
                x=np.arange(0,len(temp)/10,0.1)
                ax[i][t].plot(x, temp,color=colors[color],label=column) #On ne labellise que la première pour éviter la redondance
                ax[i][t].set_title(column)
                ax[i][t].set_ylim(y_min,y_max)
                color+=1
                if color==len(colors):
                    color=0
            else:
                for j in range(0,len(value)):
                    if j==0:
                        label=True
                    temp=df[value[j]:value[j]+time_window]
                    x=np.arange(0,len(temp)/10,0.1)
                    if label :
                        ax[i][t].plot(x, temp[column],color=colors[color],label=column) #On ne labellise que la première pour éviter la redondanvce
                        label= not label
                        ax[i][t].set_title(column)
                    else:
                        ax[i][t].plot(x, temp[column],color=colors[color])
                ax[i][t].set_ylim(y_min,y_max)
                color+=1
                if color==len(colors):
                    color=0
        #plt.legend([column])

        # set legend position
        fig.legend(bbox_to_anchor=(1.4, 0.65))

        # set spacing to subplots
        fig.tight_layout() 
        if save:
            if not os.path.exists(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Plots","")):
                os.mkdir(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Plots",""))
            plt.savefig(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Plots","cluster_"+str(key)))
        if show:
            plt.show() 
        else:
            plt.close()
    if save:
        print('Les courbes ont bien été sauvegardées.')


def plot_par_signal(cluster_echantillon, df, columns_, config, mean=True, show_title=True): 
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    time_window = cfg['time_window']
    n_cluster=cfg['n_cluster']
    colors = ['red', 'orange' , 'navy', 'teal', 'purple', 'lime', 'aqua', 'blue', 'maroon', 'gray', 'silver', 'fuchsia', 'green', 'yellow', 'olive']
    total=n_cluster
    dim=floor(math.sqrt(total))
    rows=total//dim+heaviside(total%dim) #Concerne l'affichage final des graphes (utilisation de la fonction subplot)
    cols=dim
    for column in columns_:
        y_min=df[column].min()
        y_max=df[column].max()
        print(f"----------------------------------- {column} ----------------------------------- ")
        fig, ax = plt.subplots(rows, cols)
        i,t=0,-1 #Ligne et colonne d'affichage (utilisation de la fonction subplot)
        color=0
        for key in range (n_cluster):
            value=cluster_echantillon[key]
            if t<cols-1:
                t+=1 #On avance à droite
            else :
                t=0
                i+=1 #On descend d'une ligne
            if mean: #On dessine le signal moyen
                temp_add=df[value[0]:value[0]+time_window][column]
                n_echantillons=len(value)
                if n_echantillons==0:
                    continue
                for j in range(1,n_echantillons):
                    temp=df[value[j]:value[j]+time_window][column]
                    temp_add=additione_df(temp_add,temp)
                temp=temp_add.apply(division,n=n_echantillons)
                x=np.arange(0,len(temp)/10,0.1)
                ax[i][t].plot(x, temp,color=colors[color],label='Cluster n°'+str(key)) #On ne labellise que la première pour éviter la redondance
                if show_title:
                    ax[i][t].set_title('Cluster n°'+str(key))
                ax[i][t].set_ylim(y_min,y_max)
                color+=1
                if color==len(colors):
                    color=0
            else: #On superpose les dessins des signaux
                for j in range(0,len(value)):
                    if j==0:
                        label=True
                    temp=df[value[j]:value[j]+time_window]
                    x=np.arange(0,len(temp)/10,0.1)
                    if label :
                        ax[i][t].plot(x, temp[column],color=colors[color],label='Cluster n°'+str(key)) #On ne labellise que la première pour éviter la redondanvce
                        label= not label
                    else:
                        ax[i][t].plot(x, temp[column],color=colors[color])
                if show_title:
                    ax[i][t].set_title('Cluster n°'+str(key))
                color+=1
                if color==len(colors):
                    color=0
        #plt.legend([column])
        
        # set legend position
        fig.legend(bbox_to_anchor=(1.4, 0.65))

        # set spacing to subplots
        fig.tight_layout() 
        plt.show() 
        

def cluster_utilisation(data_label):
    counts = pd.Series(data_label).value_counts()
    percentages = counts / len(data_label)
    return round(percentages,2)

def frequency_stability(data_label,config):
    d={}
    d['frequency']=[]
    d['stability (s)']=[]
    index_=[]
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    n_cluster=cfg['n_cluster']
    c_u=cluster_utilisation(data_label)
    s_r=stability_rate(data_label,config)
    for cluster in range (n_cluster):
        d['frequency'].append(str(int(c_u[cluster]*100))+" %")
        d['stability (s)'].append(s_r[cluster])
        index_.append(cluster)
    return pd.DataFrame(data=d,index=index_)

def stability_rate(data_label,config):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    n_cluster = cfg['n_cluster']
    stability_dico={}  
    for cluster in range(n_cluster):
        k=0
        occurences=0
        stability=0
        while k<len(data_label)-1:
            k+=1
            if data_label[k]==cluster:
                occurences+=1
                while data_label[k]==cluster and k<len(data_label)-1:
                    stability+=1
                    k+=1
        if occurences==0:
            stability_dico[cluster]=0
        else:
            stability_dico[cluster]=stability/(10*occurences) #10 pour avoir en secondes
    return stability_dico

def additione_df(df_a,df_a_2):
    df_a=df_a.reset_index()
    df_a_2=df_a_2.reset_index()
    output=df_a+df_a_2
    output=output.drop("index",axis=1)
    return output

def division(x,n):
    return x/n

def edge_view_to_list_without_int64(edge_view):
    output=[]
    for e in edge_view:
        (a,b)=e
        if type(a)==numpy.int64:
            a=int(a)
        if type(b)==numpy.int64:
            b=int(b)    
        output.append((a,b))
    return output

def dict_int64_to_int(hh):
    output={}
    for k, v in hh.items():
        if type(k)==numpy.int64:
            k_=int(k)
            output[k_]=v
        else:
            output[k]=v
    return output


def save_hierarchy_edges(config, df, columns_, mean=True,len_echantillon=5):
    mean_=mean
    config_file = Path(config).resolve()
    cfg = read_config(config_file)    
    n_clusters=cfg['n_cluster']
    project_path=cfg['project_path']
    transition_matrix=np.load(os.path.join(project_path,"results", "video-1",'VAME',"kmeans-"+str(n_clusters),"community","transition_matrix_video-1.npy"))
    usage=np.load(os.path.join(project_path,"results", "video-1",'VAME',"kmeans-"+str(n_clusters),"motif_usage_video-1.npy"))
    graph=graph_to_tree(usage, transition_matrix, n_cluster=n_clusters, merge_sel=1) 
    hierarchy=hierarchy_pos(graph,'Root',width=.5, vert_gap = 0.1, vert_loc = 0, xcenter = 50) 
    edge_view=graph.edges()
    edges=edge_view_to_list_without_int64(edge_view)
    hierarchy_int = dict_int64_to_int(hierarchy)
    #print(edges)
    hierarchy_edges=[hierarchy_int, edges] #Définition de hierarchy_edges
    data_label=np.load(os.path.join(project_path,"results", "video-1",'VAME',"kmeans-"+str(n_clusters),str(n_clusters)+"_km_label_video-1.npy"))
    cluster_echantillon = get_cluster_echantillon(data_label, config, len_echantillon=len_echantillon)
    #Création du dossier
    if not os.path.exists(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Hierarchy_&_Egdes","")):
        os.mkdir(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Hierarchy_&_Egdes",""))
    
    #Enregistrement de hierrarchy et edges
    #with open(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Hierarchy_&_Egdes","hierarchy_edges_"+cfg['Project']), 'w') as f:    
    with open(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),"Hierarchy_&_Egdes","hierarchy_edges"), 'w') as f:
        json.dump(hierarchy_edges, f)
        
    #Enregistrement dessins de courbes
    plot_par_cluster(cluster_echantillon, df, columns_, config, mean=mean_, show=False, save=True)
    
    
    print("Veuiller copier le lien ci-dessous et le coller en appliquant dynamic_tree.py :")
    print("")
    print(config)
    
def modify_yaml(config, parameter_name, new_value):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    with open(config, 'r') as f:
        data = yaml.safe_load(f)
    data[parameter_name] = new_value
    # Save changes to YAML file
    with open(config, 'w') as f:
        yaml.dump(data, f)
        
def silhouette_score_calculus(config,list_cluster):
    #list_cluster = [7, 8]
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    n_cluster_begining=cfg['n_cluster']
    project_path=cfg['project_path']
    #time_window=cfg['time_window']
    data_ = open_scale_clean_parquet(num_parquet,columns,  scaling=False)
    df=data_[data_['trip'].isin(trips)]
    df=df.drop("trip",axis=1)
    silhouette_avg_agg = []
    for n_clusters in list_cluster:
        modify_yaml(config, 'n_cluster', n_clusters)
        vame.pose_segmentation(config)
        Path_label=os.path.join(project_path,"results", "video-1",'VAME',"kmeans-"+str(n_clusters),str(n_clusters)+"_km_label_video-1.npy")     
        data_label=np.load(Path_label)
        length=len(data_label)
        silhouette_avg = silhouette_score(df[:length], data_label)
        silhouette_avg_agg.append(silhouette_avg)
        # Compute the silhouette scores for each sample        sample_silhouette_values = silhouette_samples(df[:100935], data_label)
        vame.visualization_for_silhouette(config, 'motif', len(df), n_clusters, sample_silhouette_values, silhouette_avg)
    modify_yaml(config, 'n_cluster', n_cluster_begining)
    return silhouette_avg_agg

def save_dataset_with_cluster(config, num_parquet, trips, columns, save=True, output=False):
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    project_path=cfg['project_path']
    n_cluster=cfg['n_cluster']
    time_window=cfg['time_window']
    Path_label=os.path.join(project_path,"results", "video-1",'VAME',"kmeans-"+str(n_cluster),str(n_cluster)+"_km_label_video-1.npy")
    data_label=np.load(Path_label)
    data_ = open_scale_clean_parquet(num_parquet,columns, scaling=False)
    df=data_[data_['trip'].isin(trips)]
    L=[]
    for i in range(time_window):
        L.append(np.nan)
    tab=np.array(L)
    # Concaténation des deux tableaux numpy en utilisant la fonction numpy.concatenate
    tableau_concatene = np.concatenate((tab, data_label))
    output = pd.concat([df.reset_index(), 'Cluster'], axis=1)
    output=output.drop("index",axis=1)
    if save:
        output.to_csv(os.path.join(cfg['project_path'],'results',"video-1",'VAME',"kmeans-"+str(n_clusters),'data_with_clusters.csv'), index=False)
    else:
        return output

def draw_color(rgba_tuple):
    #rgba_tuple=(0,0,0,1)
    # Create a figure with a single colored rectangle
    fig, ax = plt.subplots(figsize=(1,1))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=rgba_tuple))

    # Show the plot
    plt.show()
