import os
import numpy as np
import h5py
import scipy.io as scio
import _pickle as pickle

def load_data(path, adj, inp, type='flickr25k'):
    if type == 'flickr25k':
        return load_flickr25k(path, adj, inp)
    elif type == 'nus-wide':
        return load_nus_wide(path, adj, inp)
    else:
        return load_coco(path, adj, inp)


def load_flickr25k(path, adj, inp):
    images = scio.loadmat(path + 'imgList.mat')['FAll'][:].squeeze()
    all_img = np.array([name[0] for name in images])
    tags = np.array(scio.loadmat(path + 'tagList.mat')['YAll'][:])
    labels = np.array(scio.loadmat(path + 'labelList.mat')['LAll'][:])

    adj_file = scio.loadmat(adj)
    inp_file = scio.loadmat(inp)['emb']

    return all_img, tags, labels, adj_file, inp_file


def load_nus_wide(path, adj, inp):
    images = scio.loadmat(path + 'imgList.mat')['FAll']
    img_names = [images[i][0][0] for i in range(images.shape[0])]
    all_img = np.array(img_names)
    tags = np.array(h5py.File(path + 'tagList.mat')['YAll']).T
    labels = np.array(scio.loadmat(path + 'labelList.mat')['LAll'])

    adj_file = scio.loadmat(adj)
    inp_file = scio.loadmat(inp)['emb']
    return all_img, tags, labels, adj_file, inp_file


def load_coco(path, adj, inp):
    img_names = scio.loadmat(path + 'imgList.mat')['imgs']
    img_names = img_names.squeeze()
    all_img = img_names
    tags = np.array(scio.loadmat(path + 'tagList.mat')['tags'], dtype=np.float)
    labels = np.array(scio.loadmat(path + 'labelList.mat')['labels'], dtype=np.float)

    adj_file = scio.loadmat(adj)
    inp_file = scio.loadmat(inp)['emb']
    return all_img, tags, labels, adj_file, inp_file

def load_pretrain_model(path):
    return scio.loadmat(path)
