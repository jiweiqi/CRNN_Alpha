#!/usr/bin/env python
# coding: utf-8

# # Case 1: CRNN with five species and four reactions
#
# This example is reffered as the case 1 in the CRNN paper:
# * Ji, Weiqi, and Sili Deng. "Autonomous Discovery of Unknown Reaction Pathways
# from Data by Chemical Reaction Neural Network." arXiv preprint arXiv:2002.09062 (2020).
# [link](https://arxiv.org/abs/2002.09062)


# This demo code can be run in CPU within couple of minites.

import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from torch import mm
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser('case 1')
parser.add_argument('--ns', type=int, default=5)
parser.add_argument('--nr', type=int, default=4)
parser.add_argument('--n_exp', type=int, default=100)
parser.add_argument('--t_end', type=float, default=20)
parser.add_argument('--n_steps', type=int, default=101)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nepochs', type=int, default=7000)
parser.add_argument('--print_freq', type=int, default=1000)
parser.add_argument('--checkfile', type=str, default='alpha_case_1_5s4r')
parser.add_argument('--is_restart', action='store_true', default=True)
args = parser.parse_args()


class ReactorOde(object):
    def __init__(self):
        # parameters of the ODE systems and auxiliary data
        # are stored in the ReactorOde object
        self.k = [0.1, 0.2, 0.13, 0.3]
        self.dydt = np.zeros(5)

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        self.dydt[0] = -2 * self.k[0] * y[0]**2 - self.k[1] * y[0]
        self.dydt[1] = self.k[0] * y[0]**2 - self.k[3] * y[1] * y[3]
        self.dydt[2] = self.k[1] * y[0] - self.k[2] * y[2]
        self.dydt[3] = self.k[2] * y[2] - self.k[3] * y[1] * y[3]
        self.dydt[4] = self.k[3] * y[1] * y[3]

        return self.dydt


def get_solution(y0, t_end, n_steps):
    '''Use solve_ivp from scipy to solve the ODE'''
    sol = solve_ivp(ode,
                    t_span=[0, t_end],
                    y0=y0,
                    t_eval=np.linspace(0, t_end, n_steps),
                    method='BDF',
                    dense_output=False,
                    vectorized=False,
                    rtol=1e-6,
                    atol=1e-6)
    dydt = np.zeros_like(sol.y)

    for i, y in enumerate(sol.y.T):
        dydt[:, i] = ode(0, y)

    return sol.y, dydt


class CRNN_Model(nn.Module):
    '''Define CRNN'''

    def __init__(self, ns, nr):
        super(CRNN_Model, self).__init__()
        self.ns = ns
        self.nr = nr

        self.w_in = nn.Parameter(torch.zeros(ns, nr))
        self.w_b = nn.Parameter(torch.zeros(1, nr))
        self.w_out = nn.Parameter(torch.zeros(nr, ns))

        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

        # adaptive weights
        self.slope = nn.Parameter(torch.Tensor([0.1]))
        self.n = torch.Tensor([10.0])

    def forward(self, input):

        out = mm(input, self.w_in.abs()) + self.w_b
        out = torch.exp(out * self.slope * self.n)
        out = mm(out, self.w_out)

        return out

    def nslope(self):

        return self.slope.item() * self.n.item()

    def show(self):

        np.set_printoptions(precision=3, suppress=True)

        nslope = self.nslope()

        print('nslope = {:.2f}'.format(nslope))

        print('w_in')

        print(self.w_in.abs().T.data.numpy() * nslope)

        print('w_b')

        w_out_max = self.w_out.abs().max(dim=1).values

        scaled_k = torch.exp(self.w_b*nslope) * w_out_max

        print(scaled_k.detach().numpy())

        print('w_out')

        scaled_w_out = self.w_out.T / w_out_max

        print(scaled_w_out.detach().numpy().T)

    def pnorm(self):
        '''Return the L2 norm of CRNN model parameters'''

        total_norm = 0

        for param in self.parameters():
            param_norm = param.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def pgradnorm(self):
        '''Return the L2 norm of the gradient of CRNN model parameters'''

        total_norm_grad = 0

        for param in self.parameters():
            param_norm_grad = param.grad.data.norm(2)
            total_norm_grad += param_norm_grad.item() ** 2
        total_norm_grad = total_norm_grad ** (1. / 2)

        return total_norm_grad


class ReactorOdeNN(object):
    def __init__(self, model):

        self.model = model

    def __call__(self, t, y):

        with torch.no_grad():
            y_log = torch.log(torch.Tensor(y).clamp(1e-6)).view(-1, 5)
            dydt = self.model(y_log)[0]

        return dydt


class ODEDataSet(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def monitor_train(label, pred, loss_list, fname=None):
    fig = plt.figure(figsize=(12, 7))
    for i in range(5):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(label[:, i], pred[:, i].data.numpy(), 'o')
        ax.set_xlabel('Label')
        ax.set_ylabel('Pred')
    ax = fig.add_subplot(2, 3, 5+1)
    ax.plot(loss_list['train'], '-', lw=2, label='train')
    ax.plot(loss_list['test'], '--', lw=2, label='test')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=120)
    plt.show()


if __name__ == "__main__":

    checkfile = './log/' + args.checkfile
    np.random.seed(0)
    torch.manual_seed(0)

    makedirs('log')
    makedirs('fig')

    # Generate Datasets

    ode = ReactorOde()
    y_list = []
    dydt_list = []

    for i in range(args.n_exp):
        y0 = np.random.uniform(np.array([0, 0, 0, 0, 0]),
                               np.array([1, 1, 1, 1, 1]))
        y, dydt = get_solution(y0, args.t_end, args.n_steps)
        y_list.append(y.T)
        dydt_list.append(dydt.T)
    y_np = np.vstack(y_list)
    dydt_np = np.vstack(dydt_list)

    # Train CRNN Model

    crnn_model = CRNN_Model(ns=args.ns, nr=args.nr)
    optimizer = torch.optim.Adam(
        crnn_model.parameters(), lr=args.learning_rate)
    loss_func = torch.nn.MSELoss()

    X_train, X_test, Y_train, Y_test = train_test_split(y_np,
                                                        dydt_np,
                                                        test_size=0.33,
                                                        random_state=32)

    Y_max = torch.Tensor(dydt_np).abs().max(dim=0).values

    eps = 1e-12
    train_data = ODEDataSet(data=np.log(X_train+eps), label=Y_train)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=False)

    test_data = ODEDataSet(data=np.log(X_test+eps), label=Y_test)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=False)

    loss_list = {'epoch': [], 'train': [], 'test': [], 'nslope': []}

    epoch_old = 0
    if args.is_restart is True:
        checkpoint = torch.load(checkfile + '.tar')
        crnn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_old = checkpoint['epoch']
        loss_list = checkpoint['loss_list']

    for epoch in tqdm(range(args.nepochs)):
        if args.is_restart:
            if epoch < epoch_old:
                continue
        loss_train = 0
        i_sample = 0
        for i_sample, (data, label) in enumerate(train_loader):
            pred = crnn_model(data)
            loss = loss_func(pred/Y_max, label/Y_max)

            optimizer.zero_grad()
            loss.backward()
            # crnn_model.slope.grad.data *= 0
            optimizer.step()
            loss_train += loss.item()

        loss_list['train'].append(loss_train/(i_sample+1))

        with torch.no_grad():
            i_sample = 0
            loss_test = 0
            for i_sample, (data, label) in enumerate(test_loader):
                pred = crnn_model(data)
                loss = loss_func(pred/Y_max, label/Y_max)
                loss_test += loss.item()
            loss_list['test'].append(loss_test/(i_sample+1))

        loss_list['epoch'].append(epoch)
        loss_list['nslope'].append(crnn_model.nslope())

        if epoch % args.print_freq == 0:
            print("epoch: {} loss train {:.4e} test {:.4e} nslope {:.4f} pgradnorm {:.2e}".format(
                epoch, loss_list['train'][-1], loss_list['test'][-1],
                loss_list['nslope'][-1], crnn_model.pgradnorm()))

            torch.save(crnn_model, checkfile)

            torch.save({'epoch': epoch,
                        'model_state_dict': crnn_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_list': loss_list,
                        }, checkfile+'.tar')

            monitor_train(label, pred, loss_list, fname=None)

            crnn_model.show()

    # Visulize the Regression plot and Loss History
    print('\n Regression plot of five species \n')

    data = torch.from_numpy(np.log(y_np+eps)).float()
    label = torch.from_numpy(dydt_np).float()
    data = Variable(data)
    label = Variable(label)
    pred = crnn_model(data)

    monitor_train(label, pred, loss_list, fname='./fig/regression_plot')
    crnn_model.show()

    # Posterior Validation by Coupling the CRNN into ODE integration
    y0 = np.array([1, 1, 0, 0, 0])

    # here we test the CRNN for am unseen initial condition
    sol = solve_ivp(ode,
                    t_span=[0, args.t_end],
                    y0=y0,
                    t_eval=np.linspace(0, args.t_end, args.n_steps),
                    method='BDF',
                    dense_output=False,
                    vectorized=False)

    odeNN = ReactorOdeNN(crnn_model)

    solNN = solve_ivp(odeNN,
                      t_span=[0, args.t_end],
                      y0=y0,
                      t_eval=np.linspace(0, args.t_end, args.n_steps),
                      method='BDF',
                      dense_output=False,
                      vectorized=False)

    fig = plt.figure(figsize=(12, 7))
    for i in range(5):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(sol.t, sol.y[i, :], color='r', ls='solid', label='label')
        ax.plot(solNN.t, solNN.y[i, :], color='b', ls='dashed', label='crnn')
        ax.set_xlabel('Time')
        ax.set_ylabel('Conc.')
        ax.set_title('Species '+str(i+1))
        ax.legend()
    fig.tight_layout()
    plt.savefig('./fig/ode_crnn', dpi=120)
