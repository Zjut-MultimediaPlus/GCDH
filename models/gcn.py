import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import numpy as np
import math
from config import opt
from .basic_module import BasicModule


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gen_A(num_classes, t, adj_file):
    _adj = adj_file['adj']
    print(_adj.shape)
    _nums = adj_file['num']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    print(_adj.sum())
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    A = A.squeeze()
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    # print(D.shape)
    # print(A.shape)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GCN(BasicModule):

    def __init__(self, flag, hidden_dim, num_class, adj_file):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(300, 1024)
        self.gc2 = GraphConvolution(1024, hidden_dim)
        self.relu = nn.LeakyReLU(0.2)
        elif flag == "nus":
            opt.deta = 0.01
        _adj = gen_A(num_class, opt.deta, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, x, inp):
        adj = gen_adj(self.A).detach()
        inp = self.gc1(inp, adj)
        inp = self.relu(inp)
        inp = self.gc2(inp, adj)
        inp = inp.transpose(0, 1)
        x_class = torch.matmul(x, inp)
        return x_class
