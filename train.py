# -*- coding: utf-8 -*-
'''
@Time    : 2022/8/30 12:10
@Author  : KingRyu
@Email   : 1050087553@qq.com
@Project : SmaAt-UNet
@File    : train.py
@Language: Python3
'''

import os
import torch
from net_params import encoder_params, decoder_params
from net import Net
from torch.utils.data import DataLoader, DistributedSampler
from data_iter import Data_iter
from torch import optim
from early_stopping import Early_stopping
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
import numpy as np
import argparse
from encoder import Encoder
from decoder import Decoder

assert torch.cuda.device_count() > 1
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

# config
random_seed = 1997
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
n_gpus = 8

model_save_path = r'./model_save'
train_dir_path = r'/home/duan/students/KingRyu/data/radar_ds_for_dl/train'
val_dir_path = r'/home/duan/students/KingRyu/data/radar_ds_for_dl/test'
batch_size = 4
total_epoch = 100
learning_rate = 1e-4

def train():
    # 搭建网络
    encoder = Encoder(encoder_params).cuda(args.local_rank)
    decoder = Decoder(decoder_params).cuda(args.local_rank)
    net = Net(encoder, decoder)

    # 模型并行相关
    torch.distributed.init_process_group('nccl', world_size=n_gpus, rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net.cuda(args.local_rank), device_ids=[args.local_rank])

    # 判断加载已有模型和实例化优化器
    if os.path.exists(os.path.join(model_save_path, 'checkpoint.pth.tar')):
        print('loading existing model --->')
        optimizer = optim.Adam(net.parameters())
        model_dict = torch.load(os.path.join(model_save_path, 'checkpoint.pth.tar'), map_location=torch.device('cuda'))
        net.load_state_dict(model_dict['model_state'])
        optimizer.load_state_dict(model_dict['optim_state'])
        cur_epoch = model_dict['epoch'] + 1
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        cur_epoch = 0

    # 加载数据
    train_iter = Data_iter(train_dir_path)
    sampler = DistributedSampler(train_iter)
    val_iter = Data_iter(val_dir_path)
    train_loader = DataLoader(train_iter, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_iter, batch_size=batch_size, shuffle=False)

    # 实例化损失函数和训练策略
    loss_func = nn.MSELoss().cuda()
    early_stopping = Early_stopping(patience=10)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)
    # 记录1个epoch中所有train_loss
    train_losses = []
    # 记录1个epoch中所有val_loss
    val_losses = []
    # 记录所有epoch中每个epoch的平均train_loss
    avg_train_losses = []
    # 记录所有epoch中每个epoch的平均val_loss
    avg_val_losses = []
    for epoch in range(cur_epoch, total_epoch):

        sampler.set_epoch(epoch)

        #################
        # train the model
        #################
        net.train()
        tq_bar = tqdm(train_loader, leave=False, total=len(train_loader), ncols=150)
        for inputs, labels in tq_bar:
            inputs = inputs.cuda(args.local_rank)
            labels = labels.cuda(args.local_rank)
            preds = net(inputs)
            loss = loss_func(preds, labels)
            train_loss = loss.item() / batch_size
            train_losses.append(train_loss)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            tq_bar.set_postfix(
                {'train_loss': '{:.6f}'.format(train_loss),
                 'epoch': '{:02d}'.format(epoch)}
            )

        ####################
        # validate the model
        ####################
        with torch.no_grad():
            net.eval()
            tq_bar = tqdm(val_loader, leave=False, total=len(val_loader), ncols=150)
            for inputs, labels in tq_bar:
                inputs = inputs.cuda(args.local_rank)
                labels = labels.cuda(args.local_rank)
                optimizer.zero_grad()
                preds = net(inputs)
                loss = loss_func(preds, labels)
                val_loss = loss.item() / batch_size
                val_losses.append(val_loss)
                tq_bar.set_postfix(
                    {'val_loss': '{:.6f}'.format(val_loss),
                     'epoch': '{:02d}'.format(epoch)}
                )

        # 释放显存
        torch.cuda.empty_cache()

        avg_train_loss = torch.mean(train_losses)
        avg_val_loss = torch.mean(val_losses)
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)

        print_msg = (f'epoch:{epoch}/{total_epoch}' +
                     f'train_loss:{avg_train_loss:.6f}' +
                     f'val_loss:{avg_val_loss:.6f}')
        print(print_msg)

        model_dict = {
            'model_state': net.module.state_dict(),
            'optim_state': optimizer.state_dict(),
            'epoch': epoch
        }
        early_stopping(model_dict, model_save_path, avg_val_loss.item(), epoch, args.local_rank)

        if early_stopping.early_stopping:
            print('Early stopping')
            break

        pla_lr_scheduler.step(avg_val_loss)

        # 重置记录单个epoch的loss容器
        train_losses = []
        val_losses = []

if __name__ == '__main__':
    train()




