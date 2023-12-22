import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from model import DLinear
import random
import pandas as pd
import numpy as np
import heapq
import copy
import os
import sys
from models.losses import hierarchical_contrastive_loss
from data_provider.data_factory import data_provider
from tqdm import trange

def pre_train(train_data, train_loader, model, optimizer, args, target_idx):
    train_steps = len(train_loader)
    model.train()
    total_batches = len(train_loader)
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_x = batch_x.float().to(args.cuda)

        ts_l = batch_x.size(1)
        crop_l = 32
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=batch_x.size(0))

        model.setProj(True)
        out = model(take_per_row(batch_x, crop_offset + crop_eleft, 32))
        p1 = out[:, -crop_l:]
        out = model(take_per_row(batch_x, crop_offset + crop_left, 32))
        p2 = out[:, :crop_l]

        model.setProj(False)
        out = model(take_per_row(batch_x, crop_offset + crop_left, 32))
        z1 = out[:, :crop_l]
        out = model(take_per_row(batch_x, crop_offset + crop_eleft, 32))
        z2 = out[:, -crop_l:]

        loss = (hierarchical_contrastive_loss(p1,z2,temporal_unit=0) 
                + hierarchical_contrastive_loss(p2,z1,temporal_unit=0)) * 0.5
        loss.backward()
        optimizer.step()

    return loss

def train(train_data, train_loader, model, optimizer, args, target_idx):
    train_steps = len(train_loader)

    model.train()
    for i, (batch_x, batch_y) in enumerate(train_loader):

        optimizer.zero_grad()
        batch_x = batch_x.float().to(args.cuda)
        batch_y = batch_y.float().to(args.cuda)

        outputs = model(batch_x)

        f_dim = -1 
        outputs = outputs[:, -1:, f_dim:]
        batch_y = batch_y[:, -1:, target_idx].to(args.cuda)
        loss = F.mse_loss(outputs, batch_y)

        loss.backward()
        optimizer.step()

    attention = model.fs_attention
    return attention.data, loss

def test(test_data, test_loader, model, optimizer, args, target_idx):
    test_steps = len(test_loader)
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(args.cuda)
            batch_y = batch_y.float().to(args.cuda)
            outputs = model(batch_x)
            f_dim = -1 
            outputs = outputs[:, -1:, f_dim:]
            batch_y = batch_y[:, -1:, target_idx].to(args.cuda)
            loss = F.mse_loss(outputs, batch_y)
    return loss

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def GraphConstruct(target, cuda, epochs, lr, optimizername,  file, args):
    train_data, train_loader = data_provider(args, flag='train')
    test_data, test_loader = data_provider(args, flag='test')

    df_tmp = pd.read_csv(file)
    df_tmp.drop(df_tmp.columns[0], axis=1, inplace=True)
    targetidx = df_tmp.columns.get_loc(target)  

    window_size = 32
    layers = 128
    model = DLinear(window_size, layers, len(df_tmp.columns))
    
    model.to(args.cuda)
    optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)  

    model.setPretrain(True)
    pbar = trange(1, epochs+1)
    for ep in pbar:
        pretrain_loss = pre_train(train_data, train_loader, model, optimizer, args, targetidx)
        pbar.set_postfix(pretrain_loss=pretrain_loss)

    model.setPretrain(False)   
    pbar = trange(1, epochs+1)
    for ep in pbar:
        scores, train_loss = train(train_data, train_loader, model, optimizer, args, targetidx)
        model.setTest(True)
        test_loss = test(test_data, test_loader, model, optimizer, args, targetidx)
        model.setTest(False)
        pbar.set_postfix(train_loss=train_loss, test_loss=test_loss)

    s = sorted(scores.view(-1).cpu().detach().numpy(), reverse=True)
    indices = np.argsort(-1 *scores.view(-1).cpu().detach().numpy())
    
    if len(s)<=5:
        potentials = []
        for i in indices:
            if scores[i] > 1:
                potentials.append(i)
    else:
        potentials = []
        gaps = []
        for i in range(len(s)-1):
            if s[i] < 1:
                break
            gap = s[i]-s[i+1]
            gaps.append(gap)
        sortgaps = sorted(gaps, reverse=True)
        
        for i in range(0, len(gaps)):
            largestgap = sortgaps[i]
            index = gaps.index(largestgap)
            ind = -1
            if index<((len(s)-1)/2): 
                if index>0:
                    ind=index
                    break
        if ind < 0:
            ind = 0       
        potentials = indices[ : ind+1].tolist()
    edge_to_target = dict()
    for v in potentials:    
        edge_to_target[(targetidx, v)]=0
    
    return edge_to_target





