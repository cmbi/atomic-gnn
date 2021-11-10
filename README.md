## dependencies
 - python 3.8
 - numpy
 - freesasa
 - h5py
 - networkx
 - python-louvain
 - torch
 - torch-geometric
 - torch-sparse
 - torch-scatter
 - torch-cluster
 - Deeprank-GNN

## preprocessing data from bio-prodict
`python scrips/preprocess_bioprodict.py path_to_variants_parq path_to_pdb_mappings_hdf5 path_to_pdb_files path_to_pssm_files output_file_prefix --process_count 10`

## train on preprocessed hdf5 files
`python scripts/learn.py paths_to_preprocessed_hdf5_files* -f charge,sasa,wildtype,variant -a dist`

This will also create a model pth file


## test on preprocessed hdf5 files
`python scripts/test.py path_to_model_pth paths_to_preprocessed_hdf5_files* -f charge,sasa,wildtype,variant -a dist`

## calculating metrics
`python read_epoch.py results_path metrics_path`

This script needs the output HDF5 from a train or test run.
It will create a CSV table, containing metrics.
