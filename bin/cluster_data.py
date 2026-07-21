import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#This file will have functions to cluster the data via kmeans

def cluster_and_label(encoded_arr, info_df, k, return_silhouette = False):
    """This function takes in a one hot encoded array of sequences and a
    dataframe with the sequences' metadata. Then, it clusters the sequences
    via kmeans and labels each sequence's assigned cluster in the info
    dataframe.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    encoded_arr: np.array, A one hot encoded array of sequences
    info_df: pd.DataFrame, A dataframe with the sequences' metadata and a 
                column that links to the sequence's index in the encoded array
    k: int, The number of clusters to use in the kmeans run

    Returns:
    --------
    info_df: pd.DataFrame, the input dataframe with a column added
                        containing the sequence's cluster label

    """

    #First, get the cluster labels
    km = KMeans(n_clusters = k, n_init = 500)
    km = km.fit(encoded_arr)
    cluster_labels = km.labels_

    all_labels = []
    #Next, add the cluster labels to the sequence info dataframe
    for index, row in info_df.iterrows():
        encoded_ind = row['enc_index']
        cluster_label = cluster_labels[encoded_ind]
        all_labels.append(cluster_label)

    info_df['cluster_label'] = all_labels

    if return_silhouette:
        if len(np.unique(cluster_labels)) == 1:
            sil_score = None
        else:
            sil_score = silhouette_score(encoded_arr, cluster_labels)
        return info_df, sil_score
    
    return info_df

def cluster_and_label_affinity(encoded_arr, info_df):
    """This function takes in a one hot encoded array of sequences and a
    dataframe with the sequences' metadata. Then, it clusters the sequences
    via affinity propogation and labels each sequence's assigned cluster in the
    info dataframe.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    encoded_arr: np.array, A one hot encoded array of sequences
    info_df: pd.DataFrame, A dataframe with the sequences' metadata and a 
                column that links to the sequence's index in the encoded array

    Returns:
    --------
    info_df: pd.DataFrame, the input dataframe with a column added
                        containing the sequence's cluster label

    """

    #First, get the cluster labels
    ap = AffinityPropagation(max_iter= 1000, damping=0.75)
    ap = ap.fit(encoded_arr)
    cluster_labels = ap.labels_

    all_labels = []
    #Next, add the cluster labels to the sequence info dataframe
    for index, row in info_df.iterrows():
        encoded_ind = row['enc_index']
        cluster_label = cluster_labels[encoded_ind]
        all_labels.append(cluster_label)

    info_df['cluster_label'] = all_labels
    
    return info_df

def cluster_and_label_agglomerative(encoded_arr, info_df, distance_threshold):
    """This function takes in a one hot encoded array of sequences and a
    dataframe with the sequences' metadata. Then, it clusters the sequences
    via agglomerative clustering and labels each sequence's assigned cluster in
    the info dataframe.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    encoded_arr: np.array, A one hot encoded array of sequences
    info_df: pd.DataFrame, A dataframe with the sequences' metadata and a 
                column that links to the sequence's index in the encoded array
    distance_threshold: int, the distance threshold parameter to be passed to
                the agglomerative clustering algorithm

    Returns:
    --------
    info_df: pd.DataFrame, the input dataframe with a column added
                        containing the sequence's cluster label
    silhouette_score: float, the silhouette score for the current clustering
                assignments

    """

    #First, get the cluster labels
    agg = AgglomerativeClustering(n_clusters = None, linkage = 'ward',
                                  distance_threshold = distance_threshold)
    agg = agg.fit(encoded_arr)
    cluster_labels = agg.labels_

    all_labels = []
    #Next, add the cluster labels to the sequence info dataframe
    for index, row in info_df.iterrows():
        encoded_ind = row['enc_index']
        cluster_label = cluster_labels[encoded_ind]
        all_labels.append(cluster_label)

    info_df['cluster_label'] = all_labels


    
    return info_df

def pca_and_label(encoded_arr, info_df, n_components):
    """This function takes in a one hot encoded array of sequences and a
    dataframe with the sequences' metadata. Then, it performs PCA on the
    encoded array and adds columns with the sequences PCs into the dataframe.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    encoded_arr: np.array, A one hot encoded array of sequences
    info_df: pd.DataFrame, A dataframe with the sequences' metadata and a 
                column that links to the sequence's index in the encoded array
    n_components: int, The number of components to use in the PCA run

    Returns:
    --------
    info_df: pd.DataFrame, the input dataframe with columns added
                        containing the sequence's PCA components

    """
    #Get the PCA components
    my_pca = PCA(n_components = n_components)
    pca_features = my_pca.fit_transform(encoded_arr)

    #Now add the PCA features to the info dataframe
    component_list = [[] for i in range(n_components)]

    #Loop through the dataframe and add the PCA features
    for index, row in info_df.iterrows():
        if row['time_label'] == 'HXB2':
            for i in range(n_components):
                component_list[i].append(np.nan)
        else:
            encoded_ind = row['enc_index']
            for i in range(n_components):
                component_list[i].append(pca_features[encoded_ind, i])
    
    for i in range(n_components):
        info_df['pc_' + str(i+1)] = component_list[i]
    
    return info_df