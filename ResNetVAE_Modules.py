#!/usr/bin/env python
# coding: utf-8

import os 
import sys
import numpy as np
import glob

wdir = r'C:\Users\Darya\Documents\GitHub\Autoencoders'
os.chdir(wdir)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from torch.optim.lr_scheduler import StepLR
import torch_resnet_single_2_jets_7_layers as networks


from pylab import rcParams
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

import pyarrow as pa
import pyarrow.parquet as pq
import h5py

from functools import partial

import energyflow
from energyflow.emd import emd
import functools
from time import time 




class ZeroTensorPrediction(Exception): pass

def plot_img(X):
    rcParams['figure.figsize'] = 3,3
    try:
        plt.imshow(X, vmin=1.e-3, cmap='viridis', norm=LogNorm(), alpha=0.9)
        cbar = plt.colorbar()
        cbar.set_label('E')
        plt.axis('off')
        plt.show()
    except ValueError:
        return None 

class ParquetDataset(Dataset):
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None 
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jets'] = np.float32(data['X_jets'][0]) 
        # Preprocessing
        data['X_jets'] = data['X_jets'][:, 20:105, 20:105]
        data['X_jets'][data['X_jets'] < 1.e-3] = 0. # Zero-Suppression
        scaling_factors = [70, 5, 5.5]
        for i in range(3):
            if data['X_jets'][i, :, :].sum() != 0.0:
                data['X_jets'][i, :, :] = data['X_jets'][i, :, :]*scaling_factors[i] # To standardize
            
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups


def initialize(resblocks, n_channels, sch=False):
    resnet = networks.AutoEncoder(n_channels, resblocks, [16, 32, 64, 128])
    optimizer = optim.Adam(resnet.parameters(), lr=5.e-4)
    resnet.cuda()
    if sch:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    else: 
        scheduler = None 
    return resnet, optimizer, scheduler


def train_val_loader(datasets, train_cut, batch_size, random_sampler=True):
    dset = ConcatDataset([ParquetDataset(dataset) for dataset in datasets])
    idxs = np.random.permutation(len(dset))
    if random_sampler: 
        train_sampler = sampler.SubsetRandomSampler(idxs[:train_cut])
        val_sampler = sampler.SubsetRandomSampler(idxs[train_cut:])
        train_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
        val_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=val_sampler, pin_memory=True)
    else: 
        train_loader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, num_workers=0, sampler=None, pin_memory=True)
        val_loader = DataLoader(dataset=dset, batch_size=120, shuffle=False, num_workers=0, sampler=None, pin_memory=True)

    return train_loader, val_loader

def calc_loss(n_channels, X, Xreco):
    mse_total = lambda X, Xreco, n: F.mse_loss(Xreco[:, n, :, :], X[:, n, :, :])
    mse_ind = lambda X, Xreco: F.mse_loss(Xreco, X)
    if n_channels == 3:
        return sum(map(partial(mse_total, X, Xreco), range(n_channels))) 
    else: 
        return mse_ind(X, Xreco) 

def plot_orig_reco(X, Xreco, imgs, name, epoch, i, len_train_loader, loss):
    if imgs:
        print(' >> Original image:', name)
        plot_img(X.cpu().numpy().reshape(85, 85))
        if np.array_equal(np.zeros(len(Xreco.detach().cpu().numpy().flatten())), Xreco.detach().cpu().numpy().flatten()) == True:
            raise ZeroTensorPrediction
        print(' >> AE-reco image:', name)
        plot_img(Xreco.detach().cpu().numpy().reshape(85, 85))
        ind_loss = calc_loss(1, X, Xreco)
    res = '%d: (%d/%d) Total train loss:%f, Ind train loss: %f, Emax: %f, Erecomax: %f'%(epoch, i, len_train_loader, loss.item(), ind_loss.item(),  X.max(), Xreco.max())
    return res


def print_intermadiate_results(i, X, X_ind, Xreco, name, epoch, loss, len_train_loader, imgs, loss_map = False):
    if X_ind < 3:
        if imgs:
            print (plot_orig_reco(X[0], Xreco[0], imgs, name, epoch, i, len_train_loader, loss))
    else:
        for j in range(3):
            print (plot_orig_reco(X[0][j], Xreco[0][j], imgs, name, epoch, i, len_train_loader, loss))

            
def calc_emd(img1, img2): 
    return emd(img1, img2, R=1.0, norm=False, beta=1.0, measure='euclidean', coords='hadronic', return_flow=False, gdim=None, mask=False, n_iter_max=100000, periodic_phi=False, phi_col=2, empty_policy='error')

def calc_emd_reference(X, ind):
    avg = np.average(X,axis=0)
    emd_arr = []
    for i in range(X.shape[0]):
        #emd_arr = np.append(emd_arr, calc_emd(avg, sample[i]))
        emd_arr = np.append(emd_arr, calc_emd(avg, X[i]))
    return X[emd_arr.argmin()]

def calc_emd_vals(X, X_ind):
    reference = calc_emd_reference(X, X_ind)
    emd_vals = list(map(functools.partial(calc_emd, reference), X))
    return emd_vals

def eval_losses(X, Xreco, X_ind):
    ind_loss = list(map(lambda x, xreco: np.sum((x - xreco) ** 2)/float(x.shape[0] * x.shape[1]), X, Xreco))
    emd_vals = calc_emd_vals(X, X_ind)
    return ind_loss, emd_vals

def get_X(data, X_ind):
    if X_ind != 3:
        X = data['X_jets'][:, X_ind, :, :].reshape(data['X_jets'][:, X_ind, :, :].shape[0],  1, 85, 85).cuda()
    else: 
        X = data['X_jets'].cuda()
    return X

def do_eval(resnet, val_loader, epoch, X_ind, n_channels, idx=0):
    loss_, emd_vals_ind_loss = [], []
    a = time()
    monitor_step = 10
    for i, data in enumerate(val_loader):
        if i % monitor_step == 0:
            print (i)
        X = get_X(data, X_ind)
        Xreco = resnet(X)
        if X_ind != 3:
            emd_vals, ind_loss = eval_losses(X.cpu().numpy().reshape(X.shape[0], 85, 85), Xreco.detach().cpu().numpy().reshape(Xreco.shape[0], 85, 85), X_ind)
        if X_ind == 3: 
            emd_vals_ind_loss = list(map(partial(eval_losses, X.cpu().numpy()[:, X_ind-1, :, :], Xreco.detach().cpu().numpy()[:, X_ind-1, :, :]), range(0, X_ind)))
        X, Xreco = X[...,1:-1,:], Xreco[...,1:-1,:]
        loss_.append(calc_loss(n_channels, X, Xreco).tolist())
    loss_ = np.array(loss_)  
    if X_ind == 3: 
        emd_vals, ind_loss = zip(*emd_vals_ind_loss)
    s = '%d: Val loss:%f, MAE: %f, N samples: %d in %f min'%(epoch, loss_.mean(), (np.sqrt(loss_)).mean(), len(loss_), (time() -a)/60.)
    print(s)
    return loss_, emd_vals, ind_loss



def save(add_dir, obj, n, expt_name, epoch, resnet, optimizer, new_model, name, ind):
    filename = n + name + '_' + ind + '_epoch_' + str(epoch) + '.pkl'
    directory = wdir + '\%s\%s'%(add_dir, expt_name) + '\\' + filename
    if add_dir == 'MODELS':
        obj = {'model': resnet.state_dict(), 'optim': optimizer.state_dict()}
    torch.save(obj, directory)
     
    
        
def save_model(wdir, expt_name, epoch, resnet, optimizer, metrics1, metrics2, metrics3, new_model, name):
    metrics = [metrics1, metrics2, metrics3]
    names_list = ['ResNet_VAE_model_', 'ResNet_VAE_loss_', 'ResNet_VAE_emd_vals_', 'ResNet_VAE_ind_loss_']
    wdir_models = wdir + '\%s\%s'%('MODELS', expt_name)
    file_subs = [int(f.split('_')[4]) for f in listdir(wdir_models) if (isfile(join(wdir_models, f))) and (f.split('.')[1] == 'pkl') and (len(f.split('_')) > 6)]      
    if len(file_subs) == 0:
        ind = str(1)
    elif new_model: 
        ind = str(max(file_subs) + 1)
    else: 
        ind = str(max(file_subs))
    for n in names_list: 
        if n == 'ResNet_VAE_model_': 
            add_dir = 'MODELS'
            obj = None
        else: 
            add_dir = 'METRICS'
            obj = metrics[names_list.index(n) - 1]
        save(add_dir, obj, n, expt_name, epoch, resnet, optimizer, new_model, name, ind)

def train(train_loader, val_loader, resblocks, n_epochs, name, batch_size, wdir, expt_name, imgs=False):
    name_mapping = {'Tracks': 0, 'ECAL': 1, 'HCAL': 2, 'ECAL HCAL Tracks': 3}
    channels_mapping = {'Tracks': 1, 'ECAL': 1, 'HCAL': 1,  'ECAL HCAL Tracks': 3}
    X_ind, n_channels, epochs, monitor_step = name_mapping[name], channels_mapping[name], n_epochs, 10
    resnet, optimizer, scheduler = initialize(resblocks, n_channels, sch)
    resnet.train()
    a = time()
    for e in range(epochs):
        epoch = e+1
        print('>>>>>>>Epoch %d'%(epoch))
        print(">>>>>>> Training")
        for i, data in enumerate(train_loader):
            X = get_X(data, X_ind)
            Xreco = resnet(X)
            optimizer.zero_grad()
            loss = calc_loss(n_channels, X, Xreco)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if i % monitor_step == 0:
                try:
                    print_intermadiate_results(i, X, X_ind, Xreco, name, epoch, loss, len(train_loader), imgs)
                except ZeroTensorPrediction:
                    break
        print('%d: Train time:%.2f min in %d steps'%(epoch, (time() - a)/60, len(train_loader)))
        resnet.eval()
        print(">>>>>>> Validating")
        _eval, emd_vals, ind_loss = do_eval(resnet, val_loader, epoch, X_ind, n_channels)
        new_model = [True if e == 0 else False]
        save_model(wdir, expt_name, epoch, resnet, optimizer, _eval, emd_vals, ind_loss, new_model[0], name)

