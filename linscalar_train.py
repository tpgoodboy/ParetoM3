import numpy as np
import os
import random
import csv
import pickle
from tqdm import trange, tqdm

import torch
import torch.utils.data
from torch.autograd import Variable

from model.lenet import RegressionModel, RegressionTrain
from utils.misc import *


def train(log_dir, dataset, epochs, lr, preference, device='cuda', seed=42):
    
    log_path = os.path.join(log_dir, f"{dataset}_epoch{epochs}_lr{lr}_seed{seed}_pref_{preference[0]:.6f}_{preference[1]:.6f}.csv")
    f = open(log_path, "w+")
    f.close()
    with open(log_path, 'a', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'loss1', 'loss2', 'acc1', 'acc2'])
    
    set_random_seed(seed)
    
    device = torch.device(device)
    
    n_tasks = 2
    print("Preference Vector = {}".format(preference))
    preference = preference[::-1] / preference.sum()

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
    model = model.to(device)
    # ---------***---------

    # DEFINE OPTIMIZERS
    # -----------------
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.)

    # ---------***---------
    _, n_params = getNumParams(model.parameters())
    print(f"# params={n_params}")
    # ---------***---------

    # TRAIN
    # -----
    preference = torch.from_numpy(preference).to(device).reshape(-1)
    
    for t in trange(epochs):
        
        train_loss = []

        model.train()   
        for (it, batch) in enumerate(train_loader):
            X = batch[0]
            ts = batch[1]
            X = X.to(device)
            ts = ts.to(device)

            optimizer.zero_grad()
            task_loss = model(X, ts)

            weighted_loss = torch.sum(preference * task_loss)
            weighted_loss.backward()
            optimizer.step()
            
            train_loss.append(task_loss.detach().cpu().numpy())
        
        train_loss = np.stack(train_loss).mean(0)

        model.eval()
        with torch.no_grad():
            correct1_train = 0
            correct2_train = 0

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
            csv_writer.writerow([t, train_loss[0], train_loss[1], test_acc[0], test_acc[1]])
        
        tqdm.write(f"Epoch {t+1}/{epochs}, task_1 loss = {train_loss[0]:.4f}, task_2 loss = {train_loss[1]:.4f}, task_1 acc = {test_acc[0]:.4f}, task_2 acc = {test_acc[1]:.4f}.")

    return


if __name__ == '__main__':
    
    log_dir = 'linscalar_results'
    os.makedirs(log_dir, exist_ok=True)

    seed = 42
    K = 5
    
    preferences = circle_points(K, min_angle=0, max_angle=np.pi/2)
    
    for k in range(preferences.shape[0]):
        train(log_dir, 'mnist', epochs=100, lr=1.e-3, preference=preferences[k], device='cuda:0', seed=seed)
        train(log_dir, 'fashion', epochs=100, lr=1.e-3, preference=preferences[k], device='cuda:0', seed=seed)
        train(log_dir, 'fashion_and_mnist', epochs=100, lr=1.e-3, preference=preferences[k], device='cuda:0', seed=seed)
    