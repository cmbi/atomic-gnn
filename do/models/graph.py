import numpy
import logging

from torch import tensor
from torch_geometric.data import Data as TorchData

_log = logging.getLogger(__name__)


def _identical(a1, a2):
    _log.debug("compare identical {} to {}".format(a1, a2))

    return a1.shape == a2.shape and numpy.all(a1 == a2)


class EdgeData:
    "represents the edges of one type"

    def __init__(self, edge_node_indices, edge_feature_data):
        edge_index_rows, edge_index_columns = edge_node_indices.shape
        if edge_index_columns != 2:
            raise TypeError("a edge_node_indices matrix should only have 2 columns, got {}x{}".format(edge_index_rows, edge_index_columns))

        self.edge_node_indices = edge_node_indices  # numpy array
        self.edge_feature_data = edge_feature_data  # dict of numpy arrays, keys are feature names

    def __eq__(self, other):
        return _identical(self.edge_node_indices, other.edge_node_indices) and \
               self.edge_feature_data.keys() == other.edge_feature_data.keys() and \
               all([_identical(self.edge_feature_data[key], other.edge_feature_data[key])
                    for key in self.edge_feature_data])

    def get_edge_count(self):
        row_count, column_count = self.edge_node_indices.shape
        return row_count


class GraphData:
    "represents a graph as a series of quick-operable numpy arrays"

    def __init__(self, node_names=[], node_feature_data={}, edge_data={}):
        self.node_names = node_names
        self.node_feature_data = node_feature_data  # dict of numpy arrays, keys are feature names
        self.edge_data = edge_data  # dict of EdgeData's, keys are edge type names

    def get_edge_feature_dimension(self):
        first_edge_type = list(self.edge_data.values())[0]
        dimension = 0
        for feature_data in first_edge_type.edge_feature_data.values():
            dimension += feature_data.shape[1]
        return dimension

    def get_edge_count(self, types):
        return sum([self.edge_data[t].get_edge_count() for t in types])

    def get_node_count(self):
        first_feature_matrix = list(self.node_feature_data.values())[0]
        row_count, column_count = first_feature_matrix.shape
        return row_count

    def as_torch(self, edge_type_name, internal_edge_type_name, pos_feature_name):

        x = tensor(numpy.hstack([data for data in self.node_feature_data.values()]))

        # Edges must go in both directions, so duplicate and flip.
        edge_index = tensor(numpy.vstack((self.edge_data[edge_type_name].edge_node_indices,
                                          numpy.flip(self.edge_data[edge_type_name].edge_node_indices, 1))).T).contiguous()

        internal_edge_index = tensor(numpy.vstack((self.edge_data[internal_edge_type_name].edge_node_indices,
                                                   numpy.flip(self.edge_data[internal_edge_type_name].edge_node_indices, 1))).T).contiguous()

        edge_attr = numpy.hstack([data for data in self.edge_data[edge_type_name].edge_feature_data.values()])
        edge_attr = tensor(numpy.vstack((edge_attr, edge_attr)))

        internal_edge_attr = numpy.hstack([data for data in self.edge_data[internal_edge_type_name].edge_feature_data.values()])
        internal_edge_attr = tensor(numpy.vstack((internal_edge_attr, internal_edge_attr)))

        pos = tensor(self.node_feature_data[pos_feature_name])

        return TorchData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         pos=pos, internal_edge_index=internal_edge_index, internal_edge_attr=internal_edge_attr)

    def __eq__(self, other):
        return self.node_feature_data.keys() == other.node_feature_data.keys() and \
               all([_identical(self.node_feature_data[key], other.node_feature_data[key])
                    for key in self.node_feature_data]) and \
               self.edge_data.keys() == other.edge_data.keys() and \
               all([self.edge_data[key] == other.edge_data[key]
                    for key in self.edge_data])
