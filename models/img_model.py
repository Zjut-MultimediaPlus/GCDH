import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_module import BasicModule
import torch.nn.init as init
import torchvision.models as models
from .gcn import *
from torch.nn import Parameter
from .memonger import SublinearSequential
from torch.utils.checkpoint import checkpoint


model = models.resnet50(pretrained=True)

class IMG(BasicModule):
    def __init__(self, hidden_dim, output_dim, dropout, num_class, adj_file):
        super(IMG, self).__init__()
        self.module_name = 'IMG_module'
        self.outpu_dim = output_dim
        self.drop_out = dropout
        self.features = nn.Sequential(*list(model.children())[:-1])
        # self.features = SublinearSequential(
        #     *list(self.features.children())
        # )

        # self.gc1 = GraphConvolution(300, 1024)
        # self.gc2 = GraphConvolution(1024, hidden_dim)
        # self.relu = nn.LeakyReLU(0.2)
        # _adj = gen_A(num_class, 0.4, adj_file)
        # self.A = Parameter(torch.from_numpy(_adj).float())

        self.hash_module = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Tanh()
        )


        self.weight_init()

    def weight_init(self):
        initializer = self.kaiming_init
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
        # f_x = checkpoint_sequential(self.features, x).squeeze()
        f_x = self.features(x).squeeze()
        # print(f_x.shape)
        # f_x = self.hash_module(f_x)
        # adj = gen_adj(self.A).detach()
        # inp = self.gc1(inp, adj)
        # inp = self.relu(inp)
        # inp = self.gc2(inp, adj)
        # inp = inp.transpose(0, 1)
        # x_class = torch.matmul(f_x, inp)
        x_code = self.hash_module(f_x)
        return f_x, x_code.squeeze()

    def generate_img_code(self, i):
        f_i = self.features(i).squeeze()
        f_i = self.hash_module(f_i.detach())
        # adj = gen_adj(self.A).detach()
        # inp = self.gc1(inp, adj)
        # inp = self.relu(inp)
        # inp = self.gc2(inp, adj)
        # inp = inp.transpose(0, 1)
        # i_class = torch.matmul(f_i, inp)
        return f_i.squeeze()
if __name__ == '__main__':
    model = IMG(1, 1, 1, 1, 1)