import os
import torch
from torch import nn
from torch import autograd
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset import Dataset
from config import opt
from models.dis_model import DIS
import scipy.io as scio
from models.img_model import IMG
from models.txt_model import TXT, MS_TXT
from models.gcn import GCN
from triplet_loss import *
from utils import calc_map_k, pr_curve, p_topK, Visualizer
from datasets.data_handler import load_data
import time
import pickle
# import swats
import torchvision.transforms as transforms
from torch.optim import lr_scheduler, Adam, SGD



torch.backends.cudnn.deterministic = True


def train(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(opt.device)
    opt.vis_env = '{}_{}_full'.format(opt.flag, opt.bit)
    if opt.vis_env:
        vis = Visualizer(opt.vis_env, port=opt.vis_port)


    images, tags, labels, _, inp_file = load_data(opt.data_path, opt.adj, opt.inp, type=opt.dataset)
    inp_file = torch.from_numpy(inp_file).cuda()
    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, drop_last=False,
                                    pin_memory=True, shuffle=True, num_workers=8)
    L = train_data.get_labels()
    adj = (L.t() @ L).numpy()
    num = (torch.sum(L, dim=0)).numpy()
    adj_file = {'adj': adj, 'num': num}
    L = L.cuda()
    # test
    i_query_data = Dataset(opt, images, tags, labels, test='image.query')
    i_db_data = Dataset(opt, images, tags, labels, test='image.db')
    t_query_data = Dataset(opt, images, tags, labels, test='text.query')
    t_db_data = Dataset(opt, images, tags, labels, test='text.db')

    i_query_dataloader = DataLoader(i_query_data, opt.batch_size, pin_memory=True,
                                    shuffle=False, num_workers=8)
    i_db_dataloader = DataLoader(i_db_data, opt.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=8)
    t_query_dataloader = DataLoader(t_query_data, opt.batch_size, pin_memory=True,
                                    shuffle=False, num_workers=8)
    t_db_dataloader = DataLoader(t_db_data, opt.batch_size, pin_memory=True,
                                 shuffle=False, num_workers=8)

    query_labels, db_labels = i_query_data.get_labels()

    image_model = IMG(opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).cuda()

    text_model = TXT(opt.text_dim, opt.hidden_dim, opt.bit, opt.dropout, opt.num_label, adj_file).cuda()
    # text_model = MS_TXT(opt.text_dim, opt.bit, opt.num_label, adj_file).cuda()
    gcn_model = GCN(opt.flag, opt.hidden_dim * 2, opt.num_label, adj_file).cuda()

    optimizer_img_adam = Adam(image_model.parameters(), lr=opt.lr, weight_decay=0.0005)
    optimizer_txt_adam = Adam(text_model.parameters(), lr=opt.lr, weight_decay=0.0005)
    optimizer_gcn_adam = Adam(gcn_model.parameters(), lr=opt.lr, weight_decay=0.0005)

    optimizer_img = optimizer_img_adam
    optimizer_txt = optimizer_txt_adam
    optimizer_gcn = optimizer_gcn_adam

    tri_loss = TripletLoss(reduction='sum')
    criterion = nn.BCEWithLogitsLoss()
    loss = []
    loss_quan = []
    loss_triplet = []
    loss_pred = []

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_average = 0.

    mapt2i_list = []
    mapi2t_list = []
    train_times = []
    B = torch.randn(opt.training_size, opt.bit).sign().cuda()
    H_i = torch.randn(opt.training_size, opt.bit).cuda()
    H_t = torch.randn(opt.training_size, opt.bit).cuda()
    R = torch.zeros(opt.num_label, opt.bit).cuda()
    C = B
    D = 0
    gamma = (10 ** (-6) / opt.lr) ** (1 / opt.max_epoch)


    for epoch in range(opt.max_epoch):
        print(optimizer_img)
        t1 = time.time()
        loss_qi = 0
        loss_ti = 0
        loss_qt = 0
        loss_tt = 0
        loss_c = 0
        for i, (ind, img, txt, label) in tqdm(enumerate(train_dataloader)):
            img = img.cuda()
            txt = txt.cuda()
            label = label.cuda()

            f_t, h_t = text_model(txt)
            f_i, h_i = image_model(img)
            pred = gcn_model(torch.cat((f_i, f_t), 1), inp_file)
            H_i[ind, :] = h_i.data
            H_t[ind, :] = h_t.data

            i_tri = tri_loss(opt, h_i, label, target=h_t, margin=opt.margin) + tri_loss(opt, h_i, label, target=h_i, margin=opt.margin)
            t_tri = tri_loss(opt, h_t, label, target=h_i, margin=opt.margin) + tri_loss(opt, h_t, label, target=h_t, margin=opt.margin)
            i_ql = torch.sum(torch.pow(B[ind, :] - h_i, 2))
            t_ql = torch.sum(torch.pow(B[ind, :] - h_t, 2))
            loss_quant = i_ql + t_ql
            loss_class = criterion(pred, label)
            err = opt.alpha * (i_tri + t_tri) + loss_quant + loss_class


            optimizer_txt.zero_grad()
            optimizer_img.zero_grad()
            optimizer_gcn.zero_grad()
            err.backward()
            optimizer_txt.step()
            optimizer_img.step()
            optimizer_gcn.step()

            loss_c = loss_class + loss_c
            loss_qi = i_ql + loss_qi
            loss_ti = i_tri + loss_ti
            # e_loss = err + e_loss
            loss_qt = t_ql + loss_qt
            loss_tt = t_tri + loss_tt

        for param in optimizer_img.param_groups:
            param['lr'] = opt.lr * (gamma ** (epoch + 1))

        for param in optimizer_txt.param_groups:
            param['lr'] = opt.lr * (gamma ** (epoch + 1))

        for param in optimizer_gcn.param_groups:
            param['lr'] = opt.lr * (gamma ** (epoch + 1))

        R = torch.inverse(L.t() @ L + 1 * torch.eye(opt.num_label, device=opt.device)) @ L.t() @ B
        B = (L @ R + opt.gamma * (C - D / opt.gamma) + opt.mu * (H_i + H_t)).sign()
        C = calculate_C(B + D / opt.gamma)
        D = D + opt.gamma * (B - C)
        opt.gamma = opt.rho * opt.gamma
        loss_quan.append([loss_qi.item(), loss_qt.item()])
        loss_triplet.append([loss_ti.item(), loss_tt.item()])
        loss_pred.append(loss_c.item())
        print('...epoch: %3d, img_net_loss: %3.3f' % (epoch + 1, loss_quan[-1][0] + loss_triplet[-1][0]))
        print('...epoch: %3d, txt_net_loss: %3.3f' % (epoch + 1, loss_quan[-1][1] + loss_triplet[-1][1]))
        delta_t = time.time() - t1

        if opt.vis_env:
            vis.plot('img_loss_quan', loss_quan[-1][0])
            vis.plot('img_loss_triplet', loss_triplet[-1][0])
            vis.plot('txt_loss_quan', loss_quan[-1][1])
            vis.plot('txt_loss_triplet', loss_triplet[-1][1])
            vis.plot('class_loss', loss_pred[-1])

        # validate
        if opt.valid and (epoch + 1) % opt.valid_freq == 0:
            with torch.no_grad():
                mapi2t, mapt2i = valid(image_model, text_model, inp_file, i_query_dataloader, i_db_dataloader, t_query_dataloader, t_db_dataloader,
                                       query_labels, db_labels)
            print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))

            mapi2t_list.append(mapi2t)
            mapt2i_list.append(mapt2i)
            train_times.append(delta_t)

            if opt.vis_env:
                d = {
                    'mapi2t': mapi2t,
                    'mapt2i': mapt2i
                }
                vis.plot_many(d)

            if 0.5 * (mapi2t + mapt2i) > max_average:
                max_mapi2t = mapi2t
                max_mapt2i = mapt2i
                max_average = 0.5 * (mapi2t + mapt2i)
                save_model(image_model)
                save_model(text_model)
                # print('success save!')

    print('...training procedure finish')
    if opt.valid:
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
    else:
        mapi2t, mapt2i = valid(image_model, text_model, inp_file, i_query_dataloader, i_db_dataloader, t_query_dataloader,
                               t_db_dataloader, query_labels, db_labels)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))

    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    with open(os.path.join(path, 'result.pkl'), 'wb') as f:
        pickle.dump([train_times, mapi2t_list, mapt2i_list], f)

def calculate_C(B):
    U,s,V = torch.linalg.svd(B,full_matrices=False)
    # print(U.shape) # n*r
    # print(V.shape) # r*r
    rankB = torch.linalg.matrix_rank(B)
    print(rankB)
    if rankB==B.shape[1]:
        return U @ V.t() # n*64
    elif rankB<B.shape[1]:
        # print(U.shape) # n*r
        # print(V.shape) # r*r
        # print(rankB)
        n = B.shape[0]
        I = torch.eye(n)
        QU,_ = torch.linalg.qr(I-U @ U.t()) # QU n*n
        QU = QU[:,range(B.shape[1]-rankB)]
        # print(QU.shape)
        I = torch.eye(B.shape[1])
        QV,_ = torch.linalg.qr(I-V @ V.t()) # QV r*r
        QV = QV[:,range(B.shape[1]-rankB)]
        # print(QV.shape)
        return torch.hstack(U,QU) @ torch.hstack(V,QV).t() # n*(r+r') x r*(r+r').T
    else:
        return None

def valid(image_model, text_model, inp, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
          query_labels, db_labels):
    image_model.eval()
    text_model.eval()

    qBX = generate_img_code(image_model, inp, x_query_dataloader, opt.query_size, opt.bit)
    qBY = generate_txt_code(text_model, inp, y_query_dataloader, opt.query_size, opt.bit)
    rBX = generate_img_code(image_model, inp, x_db_dataloader, opt.db_size, opt.bit)
    rBY = generate_txt_code(text_model, inp, y_db_dataloader, opt.db_size, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

    image_model.train()
    text_model.train()
    return mapi2t.item(), mapt2i.item()


def test(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(opt.device)

    images, tags, labels, adj_file, inp_file = load_data(opt.data_path, opt.adj, opt.inp, type=opt.dataset)
    inp_file = torch.from_numpy(inp_file).cuda()
    bits = [opt.bit]
    with torch.no_grad():
        for bit in bits:
            image_model = IMG(opt.hidden_dim, bit, opt.dropout, opt.num_label, adj_file).cuda()
            text_model = TXT(opt.text_dim, opt.hidden_dim, bit, opt.dropout, opt.num_label, adj_file).cuda()
            # text_model = MS_TXT(opt.text_dim, bit, opt.num_label, adj_file).cuda()
            path = 'checkpoints/' + opt.dataset + '_' + str(bit)
            # load_model(generator, path)
            load_model(image_model, path)
            load_model(text_model, path)
            # generator.eval()
            image_model.eval()
            text_model.eval()


            i_query_data = Dataset(opt, images, tags, labels, test='image.query')
            i_db_data = Dataset(opt, images, tags, labels, test='image.db')
            t_query_data = Dataset(opt, images, tags, labels, test='text.query')
            t_db_data = Dataset(opt, images, tags, labels, test='text.db')

            i_query_dataloader = DataLoader(i_query_data, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            i_db_dataloader = DataLoader(i_db_data, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            t_query_dataloader = DataLoader(t_query_data, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
            t_db_dataloader = DataLoader(t_db_data, opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

            query_labels, db_labels = i_query_data.get_labels()
            query_labels = query_labels.cuda()
            db_labels = db_labels.cuda()

            qBX = generate_img_code(image_model, inp_file, i_query_dataloader, opt.query_size, bit)
            qBY = generate_txt_code(text_model, inp_file, t_query_dataloader, opt.query_size, bit)
            rBX = generate_img_code(image_model, inp_file, i_db_dataloader, opt.db_size, bit)
            rBY = generate_txt_code(text_model, inp_file, t_db_dataloader, opt.db_size, bit)


            # p_i2t, r_i2t = pr_curve(qBX, rBY, query_labels, db_labels)
            # p_t2i, r_t2i = pr_curve(qBY, rBX, query_labels, db_labels)
            #
            # K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
            # pk_i2t = p_topK(qBX, rBY, query_labels, db_labels, K)
            # pk_t2i = p_topK(qBY, rBX, query_labels, db_labels, K)

            mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
            mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)

            print('...test MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))
            scio.savemat('./checkpoints/{}-{}-hash.mat'.format(opt.dataset, bit), {
                'Qi': qBX.cpu().numpy(), 'Qt': qBY.cpu().numpy(), 'Di': rBX.cpu().numpy(), 'Dt': rBY.cpu().numpy(),
                'retrieval_L': db_labels.cpu().numpy(), 'query_L': query_labels.cpu().numpy()
            })



def generate_img_code(model, inp, test_dataloader, num, bit):
    B = torch.zeros(num, bit).cuda()
    for i, (input_data) in tqdm(enumerate(test_dataloader)):
        input_data = input_data.cuda()
        b = model.generate_img_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B

def generate_txt_code(model, inp, test_dataloader, num, bit):
    B = torch.zeros(num, bit).cuda()

    for i, (input_data) in tqdm(enumerate(test_dataloader)):
        # input_data = input_data.cuda().unsqueeze(1).unsqueeze(-1)
        input_data = input_data.cuda()
        b = model.generate_txt_code(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def calc_loss(loss):
    l = 0.
    for v in loss.values():
        l += v[-1]
    return l


def avoid_inf(x):
    return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.max(torch.zeros_like(x), x)


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    model.save(model.module_name + '.pth', path)


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''========================::HELP::=========================          
    usage : python file.py <function> [--args=value]     
    <function> := train | test | help     
    example:
            python {0} train --lr=0.01
            python {0} help
    avaiable args (default value):'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__') and str(k) != 'parse':
            print('            {0}: {1}'.format(k, v))
    print('========================::HELP::=========================')


if __name__ == '__main__':
    import fire
    fire.Fire()


