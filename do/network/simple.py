#!/usr/bin/env python

import sys
import os
import logging

import torch
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.inits import uniform
import torch.nn.functional


_log = logging.getLogger(__name__)


class SimpleConvolutionNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, number_edge_features):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._number_edge_features = number_edge_features

        self.fc_node = torch.nn.Linear(in_channels, out_channels)
        self.fc_edge_attributes = torch.nn.Linear(number_edge_features, number_edge_features)
        self.fc_attention = torch.nn.Linear(2 * out_channels + number_edge_features, 1)
        self.reset_parameters()

    def reset_parameters(self):
        size = self._in_channels
        uniform(size, self.fc_node.weight)
        uniform(size, self.fc_attention.weight)
        uniform(size, self.fc_edge_attributes.weight)

    def forward(self, node_attributes, edge_node_indices, edge_attributes):
        if torch.any(torch.isnan(node_attributes)):
            raise ValueError("NaN node attributes")
        if torch.any(torch.isnan(edge_node_indices)):
            raise ValueError("NaN edge node indices")
        if torch.any(torch.isnan(edge_attributes)):
            raise ValueError("NaN edge attributes")

        node_indices0, node_indices1 = edge_node_indices

        node_size = len(node_attributes)

        if edge_attributes.dim() == 1:
            edge_attributes = edge_attributes.unsqueeze(-1)

        node_attributes0_output = self.fc_node(node_attributes[node_indices0])
        node_attributes1_output = self.fc_node(node_attributes[node_indices1])

        edge_attribute_output = self.fc_edge_attributes(edge_attributes)

        alpha = torch.cat([node_attributes0_output, node_attributes1_output, edge_attribute_output], dim=1)
        alpha = self.fc_attention(alpha)
        alpha = torch.nn.functional.leaky_relu(alpha)
        alpha = torch.nn.functional.softmax(alpha, dim=1)

        state = alpha * node_attributes1_output

        # allocate
        out = torch.zeros(node_size, self._out_channels).to(alpha.device)

        z = scatter_sum(state, node_indices0, dim=0, out=out)

        return z


class SimpleNet(torch.nn.Module):
    def __init__(self, input_shape, output_shape=1, input_shape_edge=1):

        super().__init__()

        # need to set edge feature count here:
        self.conv_external = SimpleConvolutionNet(input_shape, 32, input_shape_edge)

        self.conv_internal = SimpleConvolutionNet(input_shape, 32, input_shape_edge)

        self.fc1 = torch.nn.Linear(2 * 32, 128)
        self.fc2 = torch.nn.Linear(128, output_shape)

        self.dropout = 0.4

    def forward(self, data):
        act = torch.nn.functional.relu

        # EXTERNAL INTERACTION GRAPH
        external_z = act(self.conv_external(data.x, data.edge_index, data.edge_attr))

        # INTERNAL INTERACTION GRAPH
        internal_z = act(self.conv_internal(data.x, data.internal_edge_index, data.internal_edge_attr))

        # FC
        external_z = scatter_mean(external_z, data.batch, dim=0)
        internal_z = scatter_mean(internal_z, data.batch, dim=0)

        z = torch.cat([external_z, internal_z], dim=1)

        z = self.fc1(z)

        z = act(z)

        z = self.fc2(z)

        return z

