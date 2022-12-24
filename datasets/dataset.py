import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms


default_mean = (0.485, 0.456, 0.406)
default_std = (0.229, 0.224, 0.225)

mir_mean = (0.372, 0.409, 0.439)
mir_std = (0.230, 0.232, 0.242)

nus_mean = (0.392, 0.423, 0.438)
nus_std = (0.237,  0.232, 0.244)

def img_transform(flag=None):
    if flag == 'mir':
        return transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mir_mean, mir_std)])
    elif flag == 'nus':
        return transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(nus_mean, nus_std)])
    else:
        return transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(default_mean, default_std)])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, images, tags, labels, test=None, flag=None):
        self.img_dir = opt.img_dir
        self.test = test
        self.flag = flag
        self.img_transform = img_transform(opt.flag)
        np.random.seed(opt.seed)
        random_index = np.random.permutation(opt.db_size + opt.query_size)
        query_index = random_index[: opt.query_size]
        train_index = random_index[opt.query_size: opt.query_size + opt.training_size]
        retrieval_index = random_index[opt.query_size:]

        if test is None:
            train_images = images[train_index]
            train_tags = tags[train_index]
            train_labels = labels[train_index]
            self.images, self.tags, self.labels = train_images, train_tags, train_labels
        else:
            self.query_labels = labels[query_index]
            self.db_labels = labels[retrieval_index]
            if test == 'image.query':
                self.images = images[query_index]
            elif test == 'image.db':
                self.images = images[retrieval_index]
            elif test == 'text.query':
                self.tags = tags[query_index]
            elif test == 'text.db':
                self.tags = tags[retrieval_index]

    def read_img(self, item):
        image_url = os.path.join(self.img_dir, self.images[item].strip('/').strip(' '))
        image = Image.open(image_url).convert('RGB')
        image = self.img_transform(image)
        return image

    def __getitem__(self, index):
        if self.test is None:
            if self.flag == 'image':
                return (
                    index,
                    self.read_img(index),
                    torch.from_numpy(self.labels[index].astype('float32'))
                )
            elif self.flag == 'text':
                return (
                    index,
                    torch.from_numpy(self.tags[index].astype('float32')),
                    torch.from_numpy(self.labels[index].astype('float32'))
                )
            else:
                return (
                    index,
                    self.read_img(index),
                    torch.from_numpy(self.tags[index].astype('float32')),
                    torch.from_numpy(self.labels[index].astype('float32'))
                )
        elif self.test.startswith('image'):
            # return torch.from_numpy(self.images[index].astype('float32'))
            return self.read_img(index)
        elif self.test.startswith('text'):
            return torch.from_numpy(self.tags[index].astype('float32'))

    def __len__(self):
        if self.test is None:
            return len(self.labels)
        elif self.test.startswith('image'):
            return len(self.images)
        elif self.test.startswith('text'):
            return len(self.tags)

    def get_labels(self):
        if self.test is None:
            return torch.from_numpy(self.labels.astype('float32'))
        else:
            return (
                torch.from_numpy(self.query_labels.astype('float32')),
                torch.from_numpy(self.db_labels.astype('float32'))
            )