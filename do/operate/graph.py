import numpy
import logging

from torch_geometric.data import Data as TorchData

from do.models.residue import Residue
from do.models.atom import Atom
from do.models.graph import EdgeData, GraphData
from do.domain.graph import EDGETYPE_FEATURE_NAME, EDGETYPE_ENCODING, POSITION_FEATURE_NAME


_log = logging.getLogger(__name__)


def choose_data_type(value):
    if type(value) == numpy.ndarray:
        return value.dtype

    if type(value) in [tuple, list]:
        return choose_data_type(value[0])

    elif type(value) == int:
        return numpy.longlong

    elif type(value) in [float, numpy.float64]:
        return numpy.float64

    elif type(value) == bytes:
        return 'S'

    else:
        raise TypeError("Cannot handle data of type {}".format(type(value)))


def get_node_name(node):
    if type(node) == Atom:
        return "{}:{}".format(get_node_name(node.residue), node.name)

    elif type(node) == Residue:
        res_id = str(node.number)
        if node.insertion_code is not None:
            res_id += node.insertion_code

        return "{}:{}{}".format(node.chain_id, node.amino_acid.name, res_id)

    elif type(node) == str:
        return node

    else:
        return str(hash(node))


def graph_to_data(graph):
    """ converts networkx graph to GraphData object
        requirements are:
         - all nodes must have the same set of features
         - all edges must have the same set of features
         - all edges must have the type feature
    """

    node_keys = list(graph.nodes.keys())
    edge_keys = list(graph.edges.keys())

    node_count = len(node_keys)

    first_node = graph.nodes[node_keys[0]]
    first_edge = graph.edges[edge_keys[0]]

    node_feature_shapes = {key: numpy.size(value) for key, value in first_node.items()}
    edge_feature_shapes = {key: numpy.size(value) for key, value in first_edge.items() if key != EDGETYPE_FEATURE_NAME}

    node_data_types = {key: choose_data_type(value) for key, value in first_node.items()}
    edge_data_types = {key: choose_data_type(value) for key, value in first_edge.items()}

    edge_types = {graph.edges[edge_key][EDGETYPE_FEATURE_NAME] for edge_key in edge_keys}

    if len(edge_types) == 0:
        raise ValueError("No edge types found")

    # collect feature data from graph
    node_feature_data = {feature_name: numpy.full((node_count, size), numpy.nan, dtype=node_data_types[feature_name])
                         for feature_name, size in node_feature_shapes.items()}
    for feature_name in node_feature_data:
        for node_index in range(node_count):
            node_key = node_keys[node_index]
            node = graph.nodes[node_key]

            node_feature_data[feature_name][node_index] = node[feature_name]

    # collect edge indices and edge features
    edge_data = {}
    for edge_type in edge_types:
        edge_keys = []
        edges = []
        for key, edge in graph.edges.items():
            if edge[EDGETYPE_FEATURE_NAME] == edge_type:
                edge_keys.append(key)
                edges.append(edge)

        edge_count = len(edges)

        node_indices = numpy.full((edge_count, 2), numpy.nan, dtype="int64")

        feature_data = {}
        for feature_name, feature_shape in edge_feature_shapes.items():
            feature_data[feature_name] = numpy.full((edge_count, feature_shape), numpy.nan, dtype=edge_data_types[feature_name])

        for edge_index in range(edge_count):
            edge = edges[edge_index]
            edge_key = edge_keys[edge_index]
            node0_key, node1_key = edge_key

            node_indices[edge_index][0] = node_keys.index(node0_key)
            node_indices[edge_index][1] = node_keys.index(node1_key)

            for feature_name in feature_data:
                feature_data[feature_name][edge_index] = edge[feature_name]

        edge_data[edge_type.decode(EDGETYPE_ENCODING)] = EdgeData(node_indices, feature_data)

    return GraphData([get_node_name(k) for k in node_keys], node_feature_data, edge_data)


STORAGE_ENCODING = "utf_8"
STORAGEKEY_NODES = "nodes"
STORAGEKEY_EDGES = "edges"
STORAGEKEY_INTERNAL_EDGES = "internal_edges"
STORAGEKEY_NODE_FEATURES = "node_data"
STORAGEKEY_EDGE_INDICES = "edge_index"
STORAGEKEY_EDGE_FEATURES = "edge_data"
STORAGEKEY_INTERNAL_EDGE_INDICES = "internal_edge_index"
STORAGEKEY_INTERNAL_EDGE_FEATURES = "internal_edge_data"

def data_to_deeprank_hdf5(graph_data, entry_group, edge_type_chosen, internal_edge_type_chosen):
    """ writes a GraphData object to a hdf5 group in deeprank-gnn compatible format

        Args:
            edge_type_chosen (str): name of the edge type to take as deeprank edges.
            internal_edge_type_chosen (str): name of the edge type to take as deeprank internal edges.
    """

    # store the nodes
    node_names = graph_data.node_names
    entry_group.create_dataset(STORAGEKEY_NODES, data=numpy.array(list(node_names)).astype("S"))

    node_data = numpy.full((graph_data.get_node_count(), 0), numpy.nan)
    node_feature_group = entry_group.create_group(STORAGEKEY_NODE_FEATURES)
    for feature_name, feature_data in graph_data.node_feature_data.items():
        node_feature_group.create_dataset(feature_name, data=feature_data)

    # collect data for the given edge type
    if edge_type_chosen not in graph_data.edge_data:
        raise ValueError("Edge type {} not found, candidates are: {}".format(edge_type_chosen, ','.join(graph_data.edge_data.keys())))

    edge_type = graph_data.edge_data[edge_type_chosen]
    edge_names = [(node_names[index0], node_names[index1]) for index0, index1 in edge_type.edge_node_indices]

    # store edge data
    entry_group.create_dataset(STORAGEKEY_EDGES, data=numpy.array(edge_names).astype("S"))
    entry_group.create_dataset(STORAGEKEY_EDGE_INDICES, data=edge_type.edge_node_indices, dtype="int64")
    edge_feature_group = entry_group.create_group(STORAGEKEY_EDGE_FEATURES)
    for feature_name, feature_data in edge_type.edge_feature_data.items():
        edge_feature_group.create_dataset(feature_name, data=feature_data)

    # collect data for the given internal edge type
    if internal_edge_type_chosen not in graph_data.edge_data:
        raise ValueError("Internal edge type {} not found, candidates are: {}".format(internal_edge_type_chosen, ','.join(graph_data.edge_data.keys())))

    internal_edge_type = graph_data.edge_data[internal_edge_type_chosen]
    internal_edge_names = [(node_names[index0], node_names[index1]) for index0, index1 in internal_edge_type.edge_node_indices]

    # store internal edge data
    entry_group.create_dataset(STORAGEKEY_INTERNAL_EDGES, data=numpy.array(internal_edge_names).astype("S"))
    entry_group.create_dataset(STORAGEKEY_INTERNAL_EDGE_INDICES, data=internal_edge_type.edge_node_indices, dtype="int64")
    internal_edge_feature_group = entry_group.create_group(STORAGEKEY_INTERNAL_EDGE_FEATURES)
    for feature_name, feature_data in internal_edge_type.edge_feature_data.items():
        internal_edge_feature_group.create_dataset(feature_name, data=feature_data)

