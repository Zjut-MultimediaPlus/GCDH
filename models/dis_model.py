import torch
from torch import nn
from .basic_module import BasicModule
import torch.nn.init as init

class DIS(BasicModule):
    def __init__(self, input_dim, hidden_dim, hash_dim):
        super(DIS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hash_dim = hash_dim

        self.feature_dis = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim//2, 1, bias=True),
            nn.Sigmoid()
        )


        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid()
        )

        self.weight_init()


    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.feature_dis:
            initializer(m)
        for m in self.hash_dis:
            initializer(m)


    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)



    def dis_feature(self, f):
        feature_score = self.feature_dis(f.squeeze())
        return feature_score.squeeze()


    def dis_hash(self, h):
        hash_score = self.hash_dis(h.squeeze())
        return hash_score.squeeze()