import numpy as np
import os
import random
import csv
import pickle
import copy
from collections import OrderedDict
from tqdm import trange, tqdm

import torch
import torch.utils.data
from torch.autograd import Variable

from model.lenet import RegressionModel, RegressionTrain
from utils.min_norm_solver import MinNormSolver
from utils.misc import *


def train(log_dir, dataset, epochs, lr, lr_y, p_x, p_y, p_l, w_x, w_y, w_l, rho, max_rho, preference, device='cuda', seed=42):
    
    log_path = os.path.join(log_dir, f"{dataset}_epoch{epochs}_lr{lr}_lry{lr_y}_px{p_x}_py{p_y}_pl{p_l}_wx{w_x}_wy{w_y}_wl{w_l}_r{rho}_mr{max_rho}_seed{seed}_pref_{preference[0]:.6f}_{preference[1]:.6f}.csv")
    f = open(log_path, "w+")
    f.close()
    with open(log_path, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'rho', 'loss1', 'loss2', 'loss_y1', 'loss_y2', 'lamb1', 'lamb2', 'acc1', 'acc2'])
    
    set_random_seed(seed)
    
    device = torch.device(device)
    
    n_tasks = 2
    print("Preference Vector = {}".format(preference))
    preference = preference[::-1].copy()
    
    rho_step = (max_rho - rho) / (epochs - 1)

    # LOAD DATASET
    # ------------
    # MultiMNIST: multi_mnist.pickle
    if dataset == 'mnist':
        with open('data/multi_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # MultiFashionMNIST: multi_fashion.pickle
    if dataset == 'fashion':
        with open('data/multi_fashion.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    # Multi-(Fashion+MNIST): multi_fashion_and_mnist.pickle
    if dataset == 'fashion_and_mnist':
        with open('data/multi_fashion_and_mnist.pickle', 'rb') as f:
            trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000, 1, 36, 36)).float()
    trainLabel = torch.from_numpy(trainLabel).long()
    testX = torch.from_numpy(testX.reshape(20000, 1, 36, 36)).float()
    testLabel = torch.from_numpy(testLabel).long()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)
    test_set = torch.utils.data.TensorDataset(testX, testLabel)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    # ---------***---------

    # DEFINE MODEL
    # ---------------------
    model = RegressionTrain(RegressionModel(n_tasks), preference)
    model_y = RegressionTrain(RegressionModel(n_tasks), preference)

    # model = torch.compile(model) 
    # model_y = torch.compile(model_y) 

    model = model.to(device)
    model_y = model_y.to(device)
    
    model_h = OrderedDict()
    model_y_h = OrderedDict()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_h[name] = param.data.clone().detach()
    for name, param in model_y.named_parameters():
        if param.requires_grad:
            model_y_h[name] = param.data.clone().detach()
    lamb = lamb_h = np.ones(n_tasks) / n_tasks
    
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.)
    optimizer_y = torch.optim.SGD(model_y.parameters(), lr=lr_y, momentum=0.)

    # ---------***---------
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}")
    # ---------***---------
    
    # TRAIN
    # -----

    for t in trange(epochs):

        train_loss = train_loss_y = []
        lambs = []

        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            X = X.to(device)
            ts = ts.to(device)
            
            model.train()
            model_y.train()

            task_loss = model(X, ts)
            task_loss_y = model_y(X, ts)

            # renew lamb
            with torch.no_grad():
                lamb = lamb_h + (task_loss_y.data.clone().detach().cpu().numpy() - task_loss.data.clone().detach().cpu().numpy()) / p_l
                lamb = MinNormSolver._projection2simplex(lamb)
                lambs.append(lamb)
                lamb_h = (1. - w_l) * lamb_h + w_l * lamb

            optimizer.zero_grad()
            # loss = (preference[0] * task_loss[0] - preference[1] * task_loss[1])**2 / rho 
            loss = torch.from_numpy(preference).to(device) * task_loss
            loss = loss / loss.data.clone().detach().sum()
            loss = (loss * torch.log(n_tasks * loss)).sum() / rho
  
            loss += (lamb[0] * task_loss[0] + lamb[1] * task_loss[1])
            for name, param in model.named_parameters():
                if param.requires_grad:
                    loss += p_x * torch.sum((param - model_h[name])**2)
            loss.backward()
            optimizer.step()
            train_loss.append(task_loss.data.clone().detach().cpu().numpy())

            optimizer_y.zero_grad()
            loss_y = (lamb[0] * task_loss_y[0] + lamb[1] * task_loss_y[1])
            for name, param in model_y.named_parameters():
                if param.requires_grad:
                    loss_y += p_y * torch.sum((param - model_y_h[name])**2)
            loss_y.backward()
            optimizer_y.step()
            train_loss_y.append(task_loss_y.data.clone().detach().cpu().numpy())

            with torch.no_grad():
                model.eval()
                model_y.eval()

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        model_h[name] = (1. - w_x) * model_h[name] + w_x * param.data.detach()
                for name, param in model_y.named_parameters():
                    if param.requires_grad:
                        model_y_h[name] = (1. - w_y) * model_y_h[name] + w_y * param.data.detach()

        train_loss = np.stack(train_loss).mean(0)
        train_loss_y = np.stack(train_loss_y).mean(0)
        lambs = np.stack(lambs).mean(0)
        
        with torch.no_grad():
            correct1_train = 0
            correct2_train = 0

            model.eval()
            model_y.eval()
    
            for (it, batch) in enumerate(test_loader):
                X = batch[0]
                ts = batch[1]
                X = X.to(device)
                ts = ts.to(device)

                output1 = model.model(X).max(2, keepdim=True)[1][:, 0]
                output2 = model.model(X).max(2, keepdim=True)[1][:, 1]
                correct1_train += output1.eq(ts[:, 0].view_as(output1)).sum().item()
                correct2_train += output2.eq(ts[:, 1].view_as(output2)).sum().item()

            test_acc = np.array(
                [1.0 * correct1_train / len(test_loader.dataset),
                1.0 * correct2_train / len(test_loader.dataset)])

        with open(log_path, 'a', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([t, rho, train_loss[0], train_loss[1], train_loss_y[0], train_loss_y[1], lambs[0], lambs[1], test_acc[0], test_acc[1]])
        
        tqdm.write(f"Epoch {t+1}/{epochs}, rho = {rho}, task_1 loss = {train_loss[0]:.4f}, task_2 loss = {train_loss[1]:.4f}, task_1 loss_y = {train_loss_y[0]:.4f}, task_2 loss_y = {train_loss_y[1]:.4f}, lamb_1 = {lambs[0]}, lamb_2 = {lambs[1]}, task_1 acc = {test_acc[0]:.4f}, task_2 acc = {test_acc[1]:.4f}.")

        rho += rho_step
        
    return


if __name__ == '__main__':
    
    log_dir = 'paretom3_results'
    os.makedirs(log_dir, exist_ok=True)

    seed = 9999
    K = 5
    
    preferences = circle_points(K, min_angle=0, max_angle=np.pi/2)
    preferences = preferences[1:K-1]
    
    for k in range(preferences.shape[0])[1:]:
        train(log_dir, 'mnist', epochs=100, lr=1.e-3, lr_y=1.e-3, p_x=1.e-4, p_y=1.e-4, p_l=1., w_x=0.1, w_y=0.2, w_l=0.2, rho=0.1, max_rho=2.0, preference=preferences[k], device='cuda:0', seed=seed)
        train(log_dir, 'fashion', epochs=100, lr=1.e-3, lr_y=1.e-3, p_x=1.e-4, p_y=1.e-4, p_l=1., w_x=0.1, w_y=0.2, w_l=0.2, rho=0.1, max_rho=2.0, preference=preferences[k], device='cuda:0', seed=seed)
        train(log_dir, 'fashion_and_mnist', epochs=100, lr=1.e-3, lr_y=1.e-3, p_x=1.e-4, p_y=1.e-4, p_l=1., w_x=0.1, w_y=0.2, w_l=0.2, rho=0.1, max_rho=2.0, preference=preferences[k], device='cuda:0', seed=seed)