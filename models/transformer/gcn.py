import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """

        @param input: shape: [number,features]
        @param adj: shape: [number,number]
        @return:
        """
        support = self.weight(input)
        output = torch.spmm(adj.half().to_sparse_csr(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCNLayer(nn.Module):
    def __init__(self, in_features, n_hid, n_hid2, dropout):
        super(GCNLayer, self).__init__()
        self.gc1 = GraphConvolution(in_features, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid2)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.PReLU()  # you can set others

    def forward(self, x, adj):
        x = self.act(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = self.act(x)
        return x


class GCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) model.

    Args:
        in_features (int): Number of input features per node.
        n_hid (int): Number of hidden units in the first GCN layer.
        n_hid2 (int): Number of hidden units in subsequent GCN layers.
        out_features (int): Number of output features per node.
        dropout (float): Dropout rate to apply after each layer.
        layers_num (int): Number of GCN layers in the network.

    Attributes:
        layers (nn.ModuleList): List containing the GCN layers in the model.
        fc (nn.Linear): Fully connected layer mapping output features to predictions.

    Forward Args:
        x (Tensor): Input feature matrix of shape (N, in_features).
        sp_adj (Tensor): Adjacency matrix of the graph of shape (N, N).

    Forward Returns:
        Tensor: Output predictions of shape (N, out_features).
    """

    def __init__(self, in_features, n_hid, n_hid2, out_features, dropout, layers_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, n_hid, n_hid2, dropout))
        for _ in range(layers_num - 1):
            self.layers.append(GCNLayer(n_hid2, n_hid, n_hid2, dropout))
        self.fc = nn.Linear(n_hid2, out_features)

    def forward(self, x, adj):
        for m in self.layers:
            x = m(x, adj)
        x = self.fc(x)
        return x
