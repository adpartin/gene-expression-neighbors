"""
Take the k (=10) closest cell lines to every PDM sample (computed above).
Then we analyze each PDM model separately. Out of all the neighbors we found
(k X number of pdm samples in a model), we extract a total of `total_cells_needed`
cells that are the closest **to the entire pdm group**. The exact heuristic is
described in the code below. Note that the heuristic favors cell lines that appear
as close neighbors for multiple pdm samples as opposed to cell lines that appears
very close to only a small subset of pdm samples.
"""
import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, 'utils_py'))
sys.path.append(utils_path)
import utils_all as utils

import warnings
warnings.filterwarnings('ignore')

DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
METADATA_FILENAME = 'combined_metadata_2018May.txt'
OUTDIR = './cell_line_recommendations_michael'
DEFAULT_DATATYPE = np.float16
SEED=0


def init_params():
    parser = argparse.ArgumentParser(description='Find nearest cell lines to each PDM model.')
    parser.add_argument('-k', '--total_cells_per_pdm_model', dest='total_cells_per_pdm_model',
	                    type=int, default=10,
	                    help='Total number of cells to find for each PDM model.')
    return parser.parse_args()


def run(args):
    print(args)
    total_cells_per_pdm_model = args.total_cells_per_pdm_model
    utils.make_dir(path=OUTDIR)

    # Load data
    SOURCES = ['ccle', 'nci60', 'ncipdm']
    lincs = utils.CombinedRNASeqLINCS(dataset='combat', datadir=DATADIR,
                                      metadata_filename=METADATA_FILENAME, sources=SOURCES)
    
    # Extract separately the pdm data, and the cell lines data
    pdm_rna, pdm_meta = lincs.get_subset(sources=['ncipdm'])
    cell_rna, cell_meta = lincs.get_subset(sources=['ccle', 'nci60'])

    # Compute kNN
    label = 'simplified_csite'
    ref_col_name = 'Sample'
    algorithm = 'brute'
    n_neighbors = 10
    metric = 'minkowski'
    p = 2

    # Use Euclidean
    knn = utils.kNNrnaseq(df_train   = cell_rna,
                          meta_train = cell_meta,
                          df_test    = pdm_rna,
                          meta_test  = pdm_meta,
                          ref_col_name = ref_col_name,
                          label = label, 
                          n_neighbors = n_neighbors,
                          algorithm = algorithm,
                          metric='minkowski', p=2)
    knn.fit()
    knn.neighbors()
    knn.summary()

    knn_samples = knn.knn_samples.copy()
    knn_distances = knn.knn_distances.copy()

    pdm = pdm_meta[['Sample', 'core_str', label, 'descr']]
    knn_samples = pdm.merge(knn_samples, on='Sample').rename(columns={'core_str': 'pdm_model', label: 'label',
                                                                      'descr': 'passage'})
    knn_distances = pdm.merge(knn_distances, on='Sample').rename(columns={'core_str': 'pdm_model', label: 'label',
                                                                          'descr': 'passage'})

    tb_samples = pd.DataFrame(index=range(len(knn_samples['pdm_model'].unique())),
                              columns=[['pdm_model'] + [f'nbr{c+1}' for c in range(n_neighbors)]])
    tb_count = tb_samples.copy()

    total_cells_per_pdm_model = 10  # total number of cells to query for the entire PDM model
    for i, model_name in enumerate(knn_samples['pdm_model'].unique()):
        tb_samples.loc[i, 'pdm_model'] = model_name
        tb_count.loc[i, 'pdm_model'] = model_name
        
        # Extract the knn cell lines for all the samples of the current pdm model
        samples = knn_samples[knn_samples['pdm_model']==model_name]
        samples = samples[[c for c in samples.columns if 'nbr' in c]]

        # Extract the knn cell line distances for all the samples of the current pdm model
        dist = knn_distances[knn_distances['pdm_model']==model_name]
        dist = dist[[c for c in dist.columns if 'nbr' in c]]
        
        # 'tmp' contains a compiled list of the closest k cell lines to every pdm sample within the model
        tmp = pd.DataFrame({'Sample': samples.values.ravel(), 'dist': dist.values.ravel()})
        tmp['dist'] = tmp['dist'].astype('float')  # why ravel() converts the values to `object`??
        tmp['count'] = 1
        
        # the 'count' col contains the number of times each cell line is encountered
        tmp = tmp.groupby(['Sample']).agg({'dist': 'sum', 'count': 'sum'}).reset_index()
        # tmp = tmp.groupby(['Sample']).agg({'dist': 'mean', 'count': 'sum'}).reset_index()
        # tmp = tmp.groupby(['Sample']).agg({'dist': 'median', 'count': 'sum'}).reset_index()
        
        # sort rows by count and distance
        tmp = tmp.sort_values(['count', 'dist'], ascending=[False, True]).reset_index(drop=True)
        
        # Finally, retain the most 'significant' cell lines
        tb_samples.iloc[i, 1:] = tmp['Sample'][:total_cells_per_pdm_model].values
        tb_count.iloc[i, 1:] = tmp['count'][:total_cells_per_pdm_model].values
	        
    # Save results
    tb_samples.to_csv(os.path.join(OUTDIR, 'samples.csv'), index=False)
    tb_count.to_csv(os.path.join(OUTDIR, 'count.csv'), index=False)
    
    # Save raw knn tables
    knn.knn_samples.to_csv(os.path.join(OUTDIR, 'knn_samples.csv'), index=False)
    knn.knn_labels.to_csv(os.path.join(OUTDIR, 'knn_labels.csv'), index=False)
    knn.knn_distances.to_csv(os.path.join(OUTDIR, 'knn_distances.csv'), index=False)
        



def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()


