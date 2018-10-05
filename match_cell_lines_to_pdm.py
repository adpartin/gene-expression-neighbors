import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = os.path.dirname(os.path.relpath(__file__))
utils_path = os.path.abspath(os.path.join(file_path, '..', 'utils_py'))
sys.path.append(utils_path)

from pilot1_imports import *
from utils import *

DATAPATH = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
PDM_METADATA_FILENAME = 'combined_metadata_2018May.txt'
DEFAULT_DATATYPE = np.float16
SEED = 2018


def get_float_format(name):
	""" From CANDLE """
    mapping = {}
    mapping['f16'] = np.float16
    mapping['f32'] = np.float32
    mapping['f64'] = np.float64

    mapped = mapping.get(name)
    if not mapped:
        raise Exception('No mapping found for "{}"'.format(name))
    return mapped


def init_params():
    parser = argparse.ArgumentParser(description='Match k-NN cell lines to PDM.')
    # parser.add_argument('-out', '--outdir', dest='outdir',
    #                     default='.',
    #                     help='output dir to store the normalized rnaseq file')
    # parser.add_argument('-f', '--in_fname', dest='in_fname',
    #                     default='combined_rnaseq_data_lincs1000',
    #                     help='rnaseq filename to normalize')
    # parser.add_argument('-ff', '--float_format', dest='float_format',
    #                     default=argparse.SUPPRESS,
    #                     choices=['f16', 'f32', 'f64'],
    #                     help='float format of the output file')

    parser.add_argument('-n', '--n_neighbors', dest='n_neighbors',
	                    type=int,
	                    help='number of nearest neighbors to find')
    return parser.parse_args()


def run(args):
    print(args)
        dataset = args.in_fname
    outdir = args.outdir
    if ~hasattr(args, 'float_format'):
        float_format = DEFAULT_DATATYPE
    elif args.float_format in set(['f16', 'f32', 'f64']):
        get_float_format(args.float_format)


    # Load data
    # dataset = 'combined_rnaseq_data'
    # dataset = 'combined_rnaseq_data_lincs1000'
    df_rna = load_combined_rnaseq(dataset=os.path.join(DATAPATH, dataset), chunksize=2000, verbose=True)
	meta = pd.read_csv(os.path.join(DATAPATH, PDM_METADATA_FILENAME), sep='\t')
	meta = update_metadata_comb_may2018(meta)
	meta = extract_specific_datasets(meta, datasets_to_keep=datasets_to_keep)
	df_rna, meta = update_df_and_meta(df_rna, meta, on='Sample')

	# Create meta for each source
	ccle_meta = meta[meta['source']=='ccle'].reset_index(drop=True)
	nci_meta  = meta[meta['source']=='nci60'].reset_index(drop=True)
	pdm_meta  = meta[meta['source']=='ncipdm'].reset_index(drop=True)

	# Create rna for each source
	ccle_rna, ccle_meta = update_df_and_meta(df_rna, ccle_meta, on='Sample')
	nci_rna,  nci_meta  = update_df_and_meta(df_rna, nci_meta,  on='Sample')
	pdm_rna,  pdm_meta  = update_df_and_meta(df_rna, pdm_meta,  on='Sample')

	# Concat cell lines data
	cells_rna = pd.concat([nci_rna, ccle_rna], axis=0).reset_index(drop=True)
	cells_meta = pd.concat([nci_meta, ccle_meta], axis=0).reset_index(drop=True)



	# m_rna, m_meta = gen_single_rna_for_each_pdm_model(pdm_rna, pdm_meta)
	# knn_results = knn_for_single_rna_summary(cells_rna, cells_meta,
	# 										 m_rna, m_meta,
	# 										 label='simplified_csite')
	# knn_samples, knn_labels, knn_distances = knn_results

	# knn_results = knn_for_single_rna_summary(pdm_rna, pdm_meta,
	# 										 pdm_rna, pdm_meta,
	# 										 label='simplified_csite')
	# knn_samples, knn_labels, knn_distances = knn_results

	# knn_samples.to_csv('knn_samples.csv', index=False)
	# knn_labels.to_csv('knn_labels.csv', index=False)
	# knn_distances.to_csv('knn_distances.csv', index=False)


	# Compute using only euclidean
	label = 'simplified_csite'
	ref_col_name = 'Sample'
	algorithm = 'brute'
	n_neighbors = 5
	metric = 'minkowski'
	p = 2

    knn_obj = kNNrnaseq(df_train = cells_rna,
                        meta_train = cells_meta,
                        df_test = pdm_rna,
                        meta_test = pdm_meta,
                        label = label, ref_col_name = ref_col_name,
                        n_neighbors = n_neighbors, algorithm = algorithm,
                        metric = metric, p = p, metric_params = metric_params)
	        
    knn_obj.fit()
    knn_obj.neighbors()
    print(knn_obj.table_labels['match_total'].sum())
    knn_samples.to_csv('knn_samples_euclidean.csv', index=False)
	knn_labels.to_csv('knn_labels_euclidean.csv', index=False)
	knn_distances.to_csv('knn_distances_euclidean.csv', index=False)


def main():
    args = init_params()
    run(args)


if __name__ == '__main__':
    main()


