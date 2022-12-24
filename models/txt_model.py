import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from .basic_module import BasicModule
import torch.nn.init as init
from .gcn import *

SEMANTIC_EMBED = 2048


class MS_Block(nn.Module):
    def __init__(self, in_channel, out_channel, pool_level, txt_length):
        super(MS_Block, self).__init__()
        self.txt_length = txt_length
        pool_kernel = (5 * pool_level, 1)
        pool_stride = (5 * pool_level, 1)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        # resize to original size of input
        x = interpolate(x, size=(self.txt_length, 1))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MS_TXT(BasicModule):
    def __init__(self, input_dim, output_dim, num_class, adj_file):
        super(MS_TXT, self).__init__()
        self.module_name = "Multi-FusionTextNet"
        self.block1 = MS_Block(1, 1, 10, input_dim)
        self.block2 = MS_Block(1, 1, 6, input_dim)
        self.block3 = MS_Block(1, 1, 3, input_dim)
        self.block4 = MS_Block(1, 1, 2, input_dim)
        self.block5 = MS_Block(1, 1, 1, input_dim)
        self.features = nn.Sequential(
            nn.Conv2d(6, 4096, kernel_size=(input_dim, 1)),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            nn.Conv2d(4096, SEMANTIC_EMBED, kernel_size=(1, 1)),
            nn.BatchNorm2d(SEMANTIC_EMBED),
            nn.ReLU(True)
        )

        self.hash_module = nn.Sequential(
            nn.Conv2d(SEMANTIC_EMBED, output_dim, kernel_size=(1, 1)),
            nn.Tanh()
        )

        self.gc1 = GraphConvolution(300, 1024)
        self.gc2 = GraphConvolution(1024, SEMANTIC_EMBED)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_class, 0.4, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.features:
            initializer(m)
        for m in self.hash_module:
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

    def forward(self, x, inp):
        x = x.unsqueeze(1).unsqueeze(-1)
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)
        block5 = self.block5(x)
        ms = torch.cat([x, block1, block2, block3, block4, block5], dim=1)
        f_x = self.features(ms)
        # f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        # h_x = self.hash_module(f_x)
        adj = gen_adj(self.A).detach()
        inp = self.gc1(inp, adj)
        inp = self.relu(inp)
        inp = self.gc2(inp, adj)
        inp = inp.transpose(0, 1)
        x_class = torch.matmul(f_x.squeeze(), inp)
        f_x = self.hash_module(f_x).squeeze()
        return f_x, x_class

    def generate_txt_code(self, x, inp):
        x = x.unsqueeze(1).unsqueeze(-1)
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)
        block5 = self.block5(x)
        ms = torch.cat([x, block1, block2, block3, block4, block5], dim=1)
        f_x = self.features(ms)
        # f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        # h_x = self.hash_module(f_x)
        f_x = self.hash_module(f_x.detach()).squeeze()
        return f_x



class TXT(BasicModule):
    def __init__(self, nfeat, hidden_dim, output_dim, dropout, num_class, adj_file):
        super(TXT, self).__init__()
        self.drop_out = dropout
        self.module_name = 'TXT_module'
        # self.fully_c = MS_Text('text', nfeat, output_dim)
        # self.gc1 = GraphConvolution(300, 1024)
        # self.gc2 = GraphConvolution(1024, hidden_dim)
        # self.relu = nn.LeakyReLU(0.2)
        # _adj = gen_A(num_class, 0.4, adj_file)
        # self.A = Parameter(torch.from_numpy(_adj).float())
        if self.drop_out:
            self.fully_c = nn.Sequential(
                nn.Linear(nfeat, 8192, bias=True),
                nn.BatchNorm1d(8192),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(8192, 4096, True),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(4096, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True),
                nn.Dropout(0.5)
            )
        else:
            self.fully_c = nn.Sequential(
                nn.Linear(nfeat, 8192, bias=True),
                nn.BatchNorm1d(8192),
                nn.ReLU(True),
                nn.Linear(8192, 4096, True),
                nn.BatchNorm1d(4096),
                nn.ReLU(True),
                nn.Linear(4096, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(True)
            )

        self.hash_module = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Tanh()
        )
        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
        for m in self.fully_c:
            initializer(m)
        for m in self.hash_module:
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

    def forward(self, x):
        f_x = self.fully_c(x)
        # f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        # h_x = self.hash_module(f_x)
        # adj = gen_adj(self.A).detach()
        # inp = self.gc1(inp, adj)
        # inp = self.relu(inp)
        # inp = self.gc2(inp, adj)
        # inp = inp.transpose(0, 1)
        # x_class = torch.matmul(f_x, inp)
        x_code = self.hash_module(f_x)
        return f_x, x_code

    def generate_txt_code(self, x):
        f_x = self.fully_c(x)
        # f_x = f_x / torch.sqrt(torch.sum(f_x.detach() ** 2))
        h_x = self.hash_module(f_x.detach())
        # adj = gen_adj(self.A).detach()
        # inp = self.gc1(inp, adj)
        # inp = self.relu(inp)
        # inp = self.gc2(inp, adj)
        # inp = inp.transpose(0, 1)
        # x_class = torch.matmul(f_x, inp)
        return h_x
