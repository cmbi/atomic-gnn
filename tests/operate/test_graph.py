import os
from tempfile import mkstemp

from networkx import Graph
import h5py

from do.domain.graph import EDGETYPE_FEATURE_NAME, EDGETYPE_ENCODING
from do.operate.graph import graph_to_data, data_to_deeprank_hdf5


def test_make_data():
    edge_type1 = "1"
    edge_type2 = "2"

    node1 = "1"
    node2 = "2"
    node3 = "3"
    node4 = "4"

    node_feature_name = "pos"

    g = Graph()
    g.add_edge(node1, node2, dist=0.1)
    g.edges[node1, node2][EDGETYPE_FEATURE_NAME] = edge_type1.encode(EDGETYPE_ENCODING)

    g.add_edge(node2, node3, dist=0.2)
    g.edges[node2, node3][EDGETYPE_FEATURE_NAME] = edge_type1.encode(EDGETYPE_ENCODING)

    g.add_edge(node2, node4, dist=0.4)
    g.edges[node2, node4][EDGETYPE_FEATURE_NAME] = edge_type2.encode(EDGETYPE_ENCODING)

    g.nodes[node1][node_feature_name] = (0,1,0)
    g.nodes[node2][node_feature_name] = (0,1,1)
    g.nodes[node3][node_feature_name] = (0,0,1)
    g.nodes[node4][node_feature_name] = (1,1,1)

    data = graph_to_data(g)

    node_feature_data = data.node_feature_data[node_feature_name]
    assert node_feature_data.shape == (4,3), "{} shape is not 4x3".format(node_feature_data)

    assert edge_type1 in data.edge_data, "missing first edge type"
    assert edge_type2 in data.edge_data, "missing second edge type"

    edge_type1_count = len(data.edge_data[edge_type1].edge_node_indices)
    assert edge_type1_count == 2, "expected 2 edges, got {}".format(edge_type1_count)

    edge_feature_count = len(data.edge_data[edge_type1].edge_feature_data)
    assert edge_feature_count == 1, "expected 1 edge feature, got {}".format(edge_feature_count)

    tmp_file, tmp_path = mkstemp(".hdf5")
    os.close(tmp_file)
    group_name = "1"
    try:
        with h5py.File(tmp_path, 'w') as f5:
            group = f5.create_group(group_name)
            data_to_deeprank_hdf5(data, group, edge_type1, edge_type2)
    finally:
        os.remove(tmp_path)
