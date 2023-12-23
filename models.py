import math
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, withloop=False, withbn=False, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)
        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)
        if self.bias is not None:
            output = output + self.bias
        if self.bn is not None:
            output = self.bn(output)
        # output = self.activation_fn(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class LinkNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, self_loops, num_layers=2,
                 norm_mode=False, use_pairnorm='None', activation=lambda x: x):
        super(LinkNet, self).__init__()
        assert num_layers >= 2
        self.mid_layers = nn.ModuleList(
            [GraphConvolution(in_features=nfeat, out_features=nhid, withloop=self_loops, withbn=norm_mode)]
            + [GraphConvolution(in_features=nhid, out_features=nhid, withloop=self_loops, withbn=norm_mode) for i in
               range(num_layers - 2)]
        )
        self.final_layer = GraphConvolution(in_features=nhid, out_features=nclass, withloop=self_loops, withbn=norm_mode)
        self.dropout = dropout
        self.pair_norm = PairNorm(use_pairnorm, 1)
        activations_map = {'relu':torch.relu, 'tanh':torch.tanh, 'sigmoid':torch.sigmoid, 'linear':lambda x: x}
        self.activation_fn = activations_map[activation]
    def encode(self, x, adj):
        for mid_layer in self.mid_layers:
            x = mid_layer(x, adj)
            x = self.pair_norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation_fn(x)
        x = self.final_layer(x, adj)
        return x
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )

class MUTILAYERGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, self_loops, num_layers=2, norm_mode=False, use_pairnorm='None', activation=lambda x: x):
        super(MUTILAYERGCN, self).__init__()
        assert num_layers >= 2
        self.mid_layers = nn.ModuleList(
            [GraphConvolution(in_features=nfeat, out_features=nhid, withloop=self_loops, withbn=norm_mode)]
            + [GraphConvolution(in_features=nhid, out_features=nhid, withloop=self_loops, withbn=norm_mode) for i in
               range(num_layers - 2)]
        )
        self.final_layer = GraphConvolution(in_features=nhid, out_features=nclass, withloop=self_loops, withbn=norm_mode)
        self.dropout = dropout
        self.pair_norm = PairNorm(use_pairnorm, 1)
        activations_map = {'relu':torch.relu, 'tanh':torch.tanh, 'sigmoid':torch.sigmoid, 'linear':lambda x: x}
        self.activation_fn = activations_map[activation]
    def forward(self, x, adj):
        for mid_layer in self.mid_layers:
            x = mid_layer(x, adj)
            x = self.pair_norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation_fn(x)
        x = self.final_layer(x, adj)
        return F.log_softmax(x, dim=1)


class MUTILAYERGCN_PPI(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, self_loops, num_layers=2, norm_mode=False, use_pairnorm='None', activation=lambda x: x):
        super(MUTILAYERGCN_PPI, self).__init__()
        assert num_layers >= 2
        self.mid_layers = nn.ModuleList(
            [GraphConvolution(in_features=nfeat, out_features=nhid, withloop=self_loops, withbn=norm_mode)]
            + [GraphConvolution(in_features=nhid, out_features=nhid, withloop=self_loops, withbn=norm_mode) for i in
               range(num_layers - 2)]
        )
        self.final_layer = GraphConvolution(in_features=nhid, out_features=nclass, withloop=self_loops,
                                            withbn=norm_mode)
        self.dropout = dropout
        self.pair_norm = PairNorm(use_pairnorm, 1)
        activations_map = {'relu': torch.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'linear': lambda x: x}
        self.activation_fn = activations_map[activation]

    def forward(self, x, adj):
        for mid_layer in self.mid_layers:
            x = mid_layer(x, adj)
            x = self.pair_norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.activation_fn(x)
        x = self.final_layer(x, adj)
        return x
