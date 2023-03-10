import torch as t
import time
import os


class BasicModule(t.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            # print(t.load(path, map_location=lambda storage, loc: storage).keys())
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            # print(t.load(path, map_location=lambda storage, loc: storage).keys())
            self.load_state_dict(t.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if not os.path.exists(path):
            os.makedirs(path)
        t.save(self.state_dict(), os.path.join(path, name))
        # print('save success!')
        return name

    def forward(self, *input):
        pass

