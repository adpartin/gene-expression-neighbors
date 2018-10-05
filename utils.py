import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pilot1_imports import *


def gen_single_rna_for_each_pdm_model(pdm_rna, pdm_meta):
    """ Generates rna and meta dataframes such that each row corresponds to a single pdm model.
    The gene expression profiles for each model are averaged to create a single represenation of
    of the pdm model. """
    pdm_rna = pdm_rna.copy()
    pdm_meta = pdm_meta.copy()

    # Get unique pdm model names
    pdm_model_names = pdm_meta['core_str'].unique().tolist()
    print('Total number of unique pdm models: {}'.format(len(pdm_model_names)))

    # Create meta df where each row is a pdm model
    meta_cols = ['core_str', 'csite', 'ctype', 'simplified_csite', 'simplified_ctype']
    m_meta = pd.DataFrame(index=range(len(pdm_model_names)), columns=meta_cols)
    m_meta = m_meta.rename(columns={'core_str': 'Model'})
    print('m_meta.shape', m_meta.shape)

    # Create rna df where each row is model (rather than Sample)
    m_rna = pd.DataFrame(index=range(len(pdm_model_names)), columns=pdm_rna.columns)
    m_rna = m_rna.rename(columns={'Sample': 'Model'})
    print('m_rna.shape', m_rna.shape)

    # Iter over pdm models, compute the mean/median across genes, assign meta and rna info to appropriate df
    for i, pdm_name in enumerate(pdm_meta['core_str'].unique()):
        pdm_idx = pdm_meta['core_str']==pdm_name
        
        # Assign meta info for the pdm model
        tmp_meta = pdm_meta.loc[pdm_idx, meta_cols].reset_index(drop=True)
        m_meta.loc[i, 'Model'] = pdm_name
        m_meta.iloc[i, 1:] = tmp_meta.loc[0, meta_cols]
        
        # Assign rna info for the pdm model
        tmp_rna  = pdm_rna[pdm_idx]
        m_rna.loc[i, 'Model'] = pdm_name
        m_rna.iloc[i, 1:] = np.median(tmp_rna.iloc[:, 1:].values, axis=0)  # takes the median across genes(!!)

    # Rename 'Model' into 'Sample'
    # (this is required for the kNN class because the query is performed based on a specific columns name)
    m_meta.rename(columns={'Model': 'Sample'}, inplace=True)
    m_rna.rename(columns={'Model': 'Sample'}, inplace=True)

    return m_rna, m_meta, pdm_model_names


def knn_for_single_rna(cells_rna,  cells_meta,
                       single_rna, single_meta,
                       label=None, ref_col_name='Sample', 
                       dist_metrics_list=[('minkowski', 1), ('minkowski', 2), ('chebyshev', None)],
                       n_neighbors=5, algorithm='brute'):
    """ Computes knn for a single rna expression (e.g., single pdm model) using multiple distance metrics. 
    Args:
        dist_metrics_list : list of tuples; each tuple is (distance_name, distance_params), e.g., ('minkowski', 2)
        ref_col_name : reference column between single_rna and single_meta
    """
    for i, m in enumerate(dist_metrics_list):
        # print(i, m)
        metric = dist_metrics_list[i][0]
        if metric=='minkowski':
            p = dist_metrics_list[i][1]
            metric_params = None
        else:
            p = None
            metric_params = dist_metrics_list[i][1]

        # Compute kNN
        knn_obj = kNNrnaseq(df_train = cells_rna,
                            meta_train = cells_meta,
                            df_test = single_rna,
                            meta_test = single_meta,
                            label = label, ref_col_name = ref_col_name,
                            n_neighbors = n_neighbors, algorithm = algorithm,
                            metric = metric, p = p, metric_params = metric_params)
        knn_obj.fit()
        knn_obj.neighbors()

        # Write results into appropriate tables
        if i == 0:
            table_samples = knn_obj.table_samples.copy()
            table_distances = knn_obj.table_distances.copy()
            table_labels = knn_obj.table_labels.copy()
        else:
            table_samples = pd.concat([table_samples, knn_obj.table_samples], axis=0)
            table_distances = pd.concat([table_distances, knn_obj.table_distances], axis=0)
            table_labels = pd.concat([table_labels, knn_obj.table_labels], axis=0)
            
    # reset index because all indices are the same (rna expression)
    table_samples.reset_index(drop=True, inplace=True)
    table_distances.reset_index(drop=True, inplace=True)
    table_labels.reset_index(drop=True, inplace=True)
    # table_labels.drop(columns=['match_total_wgt', 'match_first'], inplace=True)

    return table_samples, table_distances, table_labels


def process_knn_multi_metrics(table_samples, table_distances, table_labels, label=None):
    """ Process knn results from knn_for_single_rna(). """
    # Flatten sample and distance tables and concat horizontally
    c0 = pd.DataFrame(table_samples.iloc[:, 1:].values.ravel()).rename(columns={0: 'Sample'})
    c1 = pd.DataFrame(table_labels.iloc[:, 3:].values.ravel()).rename(columns={0: label})

    # normalize values
    for r in range(table_distances.shape[0]):
        table_distances.iloc[r, 1:] = table_distances.iloc[r, 1:] / table_distances.iloc[r, 1:].sum()
    c2 = pd.DataFrame(table_distances.iloc[:, 1:].values.ravel()).rename(columns={0: 'dist'})

    tt = pd.concat([c0, c1, c2], axis=1)
    tt['count'] = 1
    
    # Groupby Sample, compute occurance of each nbr, average distances
    tt = tt.groupby(['Sample']).agg({label: 'unique', 'count': sum, 'dist': 'unique'}).reset_index()
    tt['dist'] = tt['dist'].map(lambda x: np.mean(x))
    tt[label] = tt[label].map(lambda x: x[0] if len(x)==1 else x)
    
    # Sort by count and then by distance
    tt = tt.sort_values(by=['count', 'dist'], ascending=[False, True]).reset_index(drop=True)    
    return tt


def knn_for_single_rna_summary(cells_rna,  cells_meta,
                               df_rna, meta,
                               label=None, ref_col_name='Sample', 
                               dist_metrics_list=[('minkowski', 1), ('minkowski', 2), ('chebyshev', None)],
                               n_neighbors=5, algorithm='brute'):
    """ ... """
    df_rna = df_rna.copy()
    meta = meta.copy()

    knn_samples = pd.DataFrame(index=range(df_rna.shape[0]),
                               columns=['Sample'] + ['nbr {}'.format(n) for n in range(n_neighbors)])

    knn_labels = pd.DataFrame(index=range(df_rna.shape[0]),
                              columns=['Sample', label, 'match_total'] + ['nbr {}'.format(n) for n in range(n_neighbors)])

    knn_distances = pd.DataFrame(index=range(df_rna.shape[0]),
                                 columns=['Sample'] + ['nbr {}'.format(n) for n in range(n_neighbors)])


    # Choose distance metrics to compute
    # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    # same metric: 'manhattan', 'l1', 'cityblock', 'minkowski' with p=1
    # same metric: 'euclidean', 'l2', 'minkowski' with p=2
    # 'cityblock', 'cosine', 'mahalanobis'
    
    for i, name in enumerate(df_rna['Sample'].values):
        if i % 100 == 0:
            print('{}/{}: {}'.format(i+1, len(df_rna['Sample']), name))
        
        # Get rnaseq of a specific pdm model
        single_rna = df_rna.iloc[i, :].to_frame().T
        single_meta = meta.iloc[i, :].to_frame().T

        knn_results = knn_for_single_rna(cells_rna,  cells_meta,
                                         single_rna, single_meta,
                                         label, ref_col_name, 
                                         dist_metrics_list,
                                         n_neighbors, algorithm)
        table_samples, table_distances, table_labels = knn_results
        
        tt = process_knn_multi_metrics(table_samples, table_distances, table_labels, label)
        
        # Assign to the final table
        knn_samples.loc[i, 'Sample'] = name
        knn_samples.iloc[i, 1:] = tt.loc[:n_neighbors-1, 'Sample'].values.tolist()  # get the sample names of the closest k neighbors

        # Assign to the final table
        knn_distances.loc[i, 'Sample'] = name
        knn_distances.iloc[i, 1:] = tt.loc[:n_neighbors-1, 'dist'].values.tolist()  # get the sample names of the closest k neighbors
        
        # Assign to the final table
        query_label = single_meta[label].values[0]
        knn_labels.loc[i, 'Sample'] = name
        knn_labels.loc[i, label] = query_label
        knn_labels.loc[i, 'match_total'] = tt.loc[:n_neighbors-1, label].tolist().count(query_label)
        knn_labels.iloc[i, 3:] = tt.loc[:n_neighbors-1, label].values.tolist()  # get the sample names of the closest k neighbors
            
    return knn_samples, knn_labels, knn_distances
        

