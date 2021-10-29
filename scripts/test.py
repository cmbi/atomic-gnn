#!/usr/bin/env python

import sys
import os
import logging

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, package_root)

from deeprank_gnn.DataSet import HDF5DataSet
from deeprank_gnn.NeuralNet import NeuralNet
from do.network.simple import SimpleNet

logging.basicConfig(filename="test-%d.log" % os.getpid(), filemode="w", level=logging.DEBUG)


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

    usage = "Usage: %s model_file test_hdf5_files* [-f node_features,...] [-a edge_features,...]" % sys.argv[0]

    args, kwargs = interpret_args(sys.argv[1:], usage)

    if len(args) == 0:
        raise RuntimeError("No preprocessed HDF5 files given")

    model_path = args[0]
    test_hdf5_paths = args[1:]

    node_features = kwargs.get("f", "all").split(",")
    edge_features = kwargs.get("a", "all").split(",")
    target_name = "bin_class"

    nn = NeuralNet(test_hdf5_paths, SimpleNet,
                   node_feature=node_features, edge_feature=edge_features,
                   target=target_name,
                   pretrained_model=None,
                   cluster_nodes=None,
                   batch_size=64,
                   task='class')

    nn.load_params(model_path)

    dataset = HDF5DataSet(database=test_hdf5_paths, node_feature=node_features, edge_feature=edge_features, target=target_name)
    nn.put_model_to_device(dataset, SimpleNet)

    nn.model.load_state_dict(nn.model_load_state_dict)

    nn.test(test_hdf5_paths, threshold=nn.classes[1])

    # it will generate an output file named test_data.hdf5 ..
