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
        #scaling_factors = [40, 10, 250]
        #scaling_factors = [60, 5, 360]
        #scaling_factors = [360, 5, 60]
        #scaling_factors = [90, 1, 12]
        scaling_factors = [80, 10, 30]

        for i in range(3):
            if data['X_jets'][i, :, :].sum() != 0.0:
                data['X_jets'][i, :, :] = data['X_jets'][i, :, :]*scaling_factors[i] # To standardize
            
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups


def initialize(resblocks, n_channels):
    resnet = networks.AutoEncoder(n_channels, resblocks, [16, 32, 64, 128])
    optimizer = optim.Adam(resnet.parameters(), lr=5.e-4)
    resnet.cuda()
    #scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    #return resnet, optimizer, scheduler
    return resnet, optimizer


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
    #sc_fact_arr = [4, 1, 25]
    #sc_fact_arr = [12, 1, 90]
    #sc_fact_arr = [90, 1, 12]
    sc_fact_arr = [90, 100, 12]

    #n_channels = 3
    
    #mse = lambda X, Xreco, n, sc_fact: sc_fact * F.mse_loss(Xreco[:, n, :, :], X[:, n, :, :])
    mse = lambda X, Xreco, n: F.mse_loss(Xreco[:, n, :, :], X[:, n, :, :])

    #mse = lambda X, Xreco, n, sc_fact: sc_fact * F.l1_loss(Xreco[:, n, :, :], X[:, n, :, :])

    #return sum(map(partial(mse, X, Xreco), range(n_channels), sc_fact_arr)) 
    return sum(map(partial(mse, X, Xreco), range(n_channels))) 

def plot_orig_reco(X, Xreco, imgs, name, epoch, i, len_train_loader, loss):
    if imgs:
        print(' >> Original image:', name)
        plot_img(X.cpu().numpy().reshape(85, 85))
        if np.array_equal(np.zeros(len(Xreco.detach().cpu().numpy().flatten())), Xreco.detach().cpu().numpy().flatten()) == True:
            raise ZeroTensorPrediction
        print(' >> AE-reco image:', name)
        plot_img(Xreco.detach().cpu().numpy().reshape(85, 85))
    res = '%d: (%d/%d) Train loss:%f, Emax: %f, Erecomax: %f'%(epoch, i, len_train_loader, loss.item(), X.max(), Xreco.max())
    return res


def print_intermadiate_results(i, X, X_ind, Xreco, name, epoch, loss, len_train_loader, imgs, loss_map = False):
    if X_ind < 3:
        if imgs:
            print (plot_orig_reco(X[0], Xreco[0], imgs, name, epoch, i, len_train_loader, loss))
    else:
        for i in range(3):
            print (plot_orig_reco(X[0][i], Xreco[0][0], imgs, name, epoch, i, len_train_loader, loss))

            
def calc_emd(img1, img2): 
    return emd(img1, img2, R=1.0, norm=False, beta=1.0, measure='euclidean', coords='hadronic', return_flow=False, gdim=None, mask=False, n_iter_max=100000, periodic_phi=False, phi_col=2, empty_policy='error')

def calc_emd_reference(X, ind):
    sample = X[:, ind, :, :].reshape(X.shape[0], X.shape[2], X.shape[3])
    avg = np.average(sample,axis=0)
    emd_arr = []
    for i in range(X.shape[0]):
        emd_arr = np.append(emd_arr, calc_emd(avg, sample[i]))
    return sample[emd_arr.argmin()]

def calc_emd_vals(X, X_ind):
    reference = calc_emd_reference(X, X_ind)
    X_sample = X.cpu().numpy()[:, X_ind, :, :].reshape(X.shape[0], X.shape[2], X.shape[3])
    emd_vals = list(map(functools.partial(calc_emd, reference), X_sample))
    return emd_vals

def do_eval(resnet, val_loader, epoch, X_ind, n_channels, idx=0):
    loss_ = []
    emd_arr = []
    ind_loss = []
    a = time()
    monitor_step = 10
    for i, data in enumerate(val_loader):
        if i % monitor_step == 0:
            print (i)
        if X_ind != 3:
            X = data['X_jets'][:, X_ind, :, :].reshape(data['X_jets'][:, X_ind, :, :].shape[0],  1, 85, 85).cuda()
            emd_vals = calc_emd_vals(X, X_ind)
        else: 
            X = data['X_jets'].cuda()
            emd_vals = []
            for i in range(3):
                emd_vals.append(calc_emd_vals(X, i))
        Xreco = resnet(X)
        if X_ind != 3:
            ind_loss = list(map(lambda x, xreco: np.sum((x - xreco) ** 2)/float(x.shape[0] * x.shape[1]), X.cpu().numpy()[:, X_ind, :, :], Xreco.detach().cpu().numpy()[:, X_ind, :, :]))
        else: 
            for i in range(3): 
                ind_loss.append(list(map(lambda x, xreco: np.sum((x - xreco) ** 2)/float(x.shape[0] * x.shape[1]), X.cpu().numpy()[:, i, :, :], Xreco.detach().cpu().numpy()[:, i, :, :])))
        X = X[...,1:-1,:]
        Xreco = Xreco[...,1:-1,:]
        losses = calc_loss(n_channels, X, Xreco)
        loss_.append(losses.tolist())
    loss_ = np.array(loss_)  
    s = '%d: Val loss:%f, MAE: %f, N samples: %d in %f min'%(epoch, loss_.mean(), (np.sqrt(loss_)).mean(), len(loss_), (time() -a)/60.)
    print(s)
    return loss_, emd_vals, ind_loss



def save(add_dir, obj, n, expt_name, epoch, resnet, optimizer, new_model, name, ind):
    filename = n + name + '_' + ind + '_epoch_' + str(epoch) + '.pkl'
    directory = wdir + '\%s\%s'%(add_dir, expt_name) + filename
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
    X_ind = name_mapping[name]
    n_channels = channels_mapping[name]
    resnet, optimizer = initialize(resblocks, n_channels)
    resnet.cuda()
    epochs = n_epochs
    monitor_step = 10
    resnet.train()
    metrics = []
    a = time()
    for e in range(epochs):
        epoch = e+1
        s = '>> Epoch %d <<<<<<<<'%(epoch)
        print(s)

        print(">> Training <<<<<<<<")
        #now = time.time()
        for i, data in enumerate(train_loader):
            if X_ind != 3:
                X = data['X_jets'][:, X_ind, :, :].reshape(data['X_jets'][:, X_ind, :, :].shape[0], 1, 85, 85).cuda()
            else: 
                X = data['X_jets'].cuda()
            optimizer.zero_grad()
            Xreco = resnet(X)
            loss = calc_loss(n_channels, X, Xreco)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            if i % monitor_step == 0:
                #if i!= 0:
                try:
                    print_intermadiate_results(i, X, X_ind, Xreco, name, epoch, loss, len(train_loader), imgs)
                except ZeroTensorPrediction:
                    break
                # else: 
                #     print_intermadiate_results(i, X, X_ind, Xreco, name, epoch, loss, len(train_loader), imgs)
        s = '%d: Train time:%.2f min in %d steps'%(epoch, (time() - a)/60, len(train_loader))
        print(s)
        resnet.eval()
        print(">> Validation: Good samples <<<<<<<<")
        _eval, emd_vals, ind_loss = do_eval(resnet, val_loader, epoch, X_ind, n_channels)
        if e == 0: 
            new_model= True
            print ('new model')
        else: 
            new_model= False
        save_model(wdir, expt_name, epoch, resnet, optimizer, _eval, emd_vals, ind_loss, new_model, name)
        #metrics.append(_eval)

