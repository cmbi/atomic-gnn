#!/usr/bin/env python

import sys
import os
import logging

import torch

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, package_root)

from deeprank_gnn.NeuralNet import NeuralNet
from do.network.simple import SimpleNet

logging.basicConfig(filename="learn-%d.log" % os.getpid(), filemode="w", level=logging.DEBUG)

_log = logging.getLogger(__name__)

def interpret_args(args, usage):
    """ Convert a list of commandline arguments into a set of positional and keyword arguments.
        Args (list of str): the commandline arguments
        Returns: (tuple(list of str, dict of str)): the positional and keyword arguments
    """

    if len(args) == 0:
        print(usage)
        sys.exit(1)

    if "--help" in args or "-h" in args:
        print(usage)
        sys.exit(0)

    positional_args = []
    kwargs = {}
    i = 0
    while i < len(args):

        if args[i].startswith("--"):
            key = args[i][2:]

            i += 1
            kwargs[key] = args[i]

        elif args[i].startswith("-"):
            key = args[i][1:2]

            if len(args[i]) > 2:
                kwargs[key] = args[i][2:]
            else:
                i += 1
                kwargs[key] = args[i]
        else:
            positional_args.append(args[i])

        i += 1

    return (positional_args, kwargs)


if __name__ == "__main__":

    usage = "Usage: %s [-e epoch_count] [-l learn_rate] [-f node_features, ...] [-a edge_features, ...] *preprocessed_hdf5_files" % sys.argv[0]

    args, kwargs = interpret_args(sys.argv[1:], usage)

    if len(args) == 0:
        raise RuntimeError("No preprocessed HDF5 files given")

    hdf5_paths = args

    epoch_count = int(kwargs.get('e', 250))
    learn_rate = float(kwargs.get('l', 0.001))

    node_features = kwargs.get("f", "all").split(",")
    edge_features = kwargs.get("a", "all").split(",")

    _log.info("training with input files {} and node features {} and edge features {}"
              .format(hdf5_paths, node_features, edge_features))

    torch.manual_seed(7432609565498794389263420134)

    nn = NeuralNet(database=hdf5_paths, Net=SimpleNet,
                   cluster_nodes="louvain",
                   node_feature=node_features, edge_feature=edge_features,
                   target='bin_class', lr=learn_rate,
                   batch_size=64,
                   percent=[0.8, 0.2],
                   task='class')

    nn.train(nepoch=epoch_count, validate=True, save_epoch="all")

