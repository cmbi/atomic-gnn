import os
from tempfile import mkdtemp
from shutil import rmtree

import logging
import h5py
from deeprank_gnn.DataSet import HDF5DataSet
from deeprank_gnn.NeuralNet import NeuralNet
import torch
from torch_scatter import scatter_mean

from do.models.variant import PdbVariantSelection, VariantClass
from do.variant_atomic_graph import add_as_graph
from do.domain.amino_acid import histidine, alanine, leucine
from do.network.simple import SimpleNet


_log = logging.getLogger(__name__)


node_features = ['charge', 'element', 'sasa', 'pos', 'wildtype', 'variant']
edge_features = ['dist', 'coulomb', 'vanderwaals']


def test_loadable():

    tmp_dir = mkdtemp()
    try:
        pdb_path = "tests/data/101M.pdb"

        variants = [PdbVariantSelection(pdb_path, "A", 36, None, histidine, alanine, variant_class=VariantClass.PATHOGENIC),
                    PdbVariantSelection(pdb_path, "A", 11, None, leucine, alanine, variant_class=VariantClass.BENIGN)]

        hdf5_path = os.path.join(tmp_dir, "out.hdf5")

        for variant in variants:
            add_as_graph(variant, hdf5_path)

        with h5py.File(hdf5_path, 'r') as f:
            entry_name = list(f.keys())[0]

            edge_indices = f["{}/edge_index".format(entry_name)][()]
            assert edge_indices.shape[1] == 2, "edge indices have shape {}".format(edge_indices.shape)

            internal_edge_indices = f["{}/internal_edge_index".format(entry_name)][()]
            assert internal_edge_indices.shape[1] == 2, "internal edge indices have shape {}".format(internal_edge_indices.shape)

            edge_feature_names = f["{}/edge_data".format(entry_name)].keys()
            assert len(edge_feature_names) > 0, "no edge features"

            for edge_feature_name in edge_feature_names:
                edge_data = f["{}/edge_data/{}".format(entry_name, edge_feature_name)][()]
                assert edge_data.size > 0, "{} edge data is empty".format(edge_feature_name)

            internal_edge_feature_names = f["{}/internal_edge_data".format(entry_name)].keys()
            assert len(internal_edge_feature_names) > 0, "no internal edge features"

            for internal_edge_feature_name in internal_edge_feature_names:
                internal_edge_data = f["{}/internal_edge_data/{}".format(entry_name, internal_edge_feature_name)][()]
                assert internal_edge_data.size > 0, "{} internal edge data is empty".format(internal_edge_feature_name)

        dataset = HDF5DataSet(database=[hdf5_path], clustering_method="louvain")

        data = dataset.get(0)

        nn = NeuralNet([hdf5_path], SimpleNet,
                       cluster_nodes=None,
                       node_feature=node_features, edge_feature=edge_features,
                       target='bin_class',
                       batch_size=64,
                       percent=[0.8, 0.2],
                       task='class')

        nn.train(nepoch=1, validate=True)

    finally:
        rmtree(tmp_dir)
