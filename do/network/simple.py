#!/usr/bin/env python

import sys
import os
import logging

import torch
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import max_pool_x
import torch.nn.functional
from deeprank_gnn.community_pooling import get_preloaded_cluster, community_pooling


_log = logging.getLogger(__name__)


class SimpleConvolutionNet(torch.nn.Module):
    def __init__(self, number_node_features, number_edge_features):
        super().__init__()

        self._message_layer = torch.nn.Linear(2 * number_node_features + number_edge_features, 1)
        self._update_layer = torch.nn.Linear(number_node_features + 1, number_node_features)

    def forward(self, node_attributes, edge_node_indices, edge_attributes):
        if torch.any(torch.isnan(node_attributes)):
            raise ValueError("NaN node attributes")
        if torch.any(torch.isnan(edge_node_indices)):
            raise ValueError("NaN edge node indices")
        if torch.any(torch.isnan(edge_attributes)):
            raise ValueError("NaN edge attributes")

        node_indices0, node_indices1 = edge_node_indices

        node_attributes0_input = node_attributes[node_indices0]
        node_attributes1_input = node_attributes[node_indices1]

        message_inputs = torch.cat([node_attributes0_input, node_attributes1_input, edge_attributes], dim=1)
        messages = self._message_layer(message_inputs)
        messages = torch.nn.functional.leaky_relu(messages)
        messages = torch.nn.functional.softmax(messages, dim=1)

        # sum the messages per node
        node_messages = scatter_sum(messages, node_indices0, dim=0)

        # update the node
        update_inputs = torch.cat((node_attributes, node_messages), dim=1)
        updated_node_features = self._update_layer(update_inputs)
        updated_node_features = torch.nn.functional.leaky_relu(updated_node_features)
        updated_node_features = torch.nn.functional.softmax(updated_node_features)
        return updated_node_features


class SimpleNet(torch.nn.Module):
    def __init__(self, input_shape, output_shape=1, input_shape_edge=1):

        super().__init__()

        # need to set edge feature count here:
        self.conv_external1 = SimpleConvolutionNet(input_shape, input_shape_edge)
        self.conv_external2 = SimpleConvolutionNet(input_shape, input_shape_edge)

        self.conv_internal1 = SimpleConvolutionNet(input_shape, input_shape_edge)
        self.conv_internal2 = SimpleConvolutionNet(input_shape, input_shape_edge)

        self.fc1 = torch.nn.Linear(2 * input_shape, 128)
        self.fc2 = torch.nn.Linear(128, output_shape)

        self.dropout = 0.4

    def forward(self, data):
        act = torch.nn.functional.relu

        # EXTERNAL INTERACTION GRAPH
        external_x = self.conv_external1(data.x, data.edge_index, data.edge_attr)
        external_x = self.conv_external2(external_x, data.edge_index, data.edge_attr)

        # INTERNAL INTERACTION GRAPH
        internal_x = self.conv_internal1(data.x, data.internal_edge_index, data.internal_edge_attr)
        internal_x = self.conv_internal2(internal_x, data.internal_edge_index, data.internal_edge_attr)

        # FC
        external_z = scatter_mean(external_x, data.batch, dim=0)
        internal_z = scatter_mean(internal_x, data.batch, dim=0)

        z = torch.cat([external_z, internal_z], dim=1)

        z = self.fc1(z)

        z = act(z)

        z = self.fc2(z)

        return z

