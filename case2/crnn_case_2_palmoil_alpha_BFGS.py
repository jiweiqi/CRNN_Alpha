#!/usr/bin/env python
# coding: utf-8

# # Case 2: CRNN with six species and three reactions
#
# This example is reffered as the case 2 in the CRNN paper:
# * Ji, Weiqi, and Sili Deng. "Autonomous Discovery of Unknown Reaction Pathways
# from Data by Chemical Reaction Neural Network." arXiv preprint arXiv:2002.09062 (2020).
# [link](https://arxiv.org/abs/2002.09062)


# Initial conditions (data generation) matters a lot

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split
from torch import mm
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser('case 2')
parser.add_argument('--ns', type=int, default=6)
parser.add_argument('--nr', type=int, default=3)
parser.add_argument('--n_exp', type=int, default=300)
parser.add_argument('--t_end', type=float, default=200)
parser.add_argument('--n_steps', type=int, default=101)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--nepochs', type=int, default=100000)
parser.add_argument('--print_freq', type=int, default=500)
parser.add_argument('--checkfile', type=str, default='alpha_case_2')
parser.add_argument('--is_pruning', action='store_true', default=False)
parser.add_argument('--pruning_threshhold', type=float, default=1e-2)
parser.add_argument('--adaptive_slope', action='store_true', default=True)
parser.add_argument('--is_restart', action='store_true', default=False)
args = parser.parse_args()

Ea_scale = 10.0


class ReactorOde(object):
    def __init__(self, k, ns):
        # parameters of the ODE systems and auxiliary data
        # are stored in the ReactorOde object
        self.k = k
        self.dydt = np.zeros(ns)

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # TG,ROH,DG,MG,GL,R'CO2R

        # TG
        self.dydt[0] = - self.k[0] * y[0] * y[1]

        # ROH
        self.dydt[1] = (- self.k[0] * y[0] * y[1] -
                        self.k[1] * y[2] * y[1] - self.k[2] * y[3] * y[1])

        # DG
        self.dydt[2] = self.k[0] * y[0] * y[1] - self.k[1] * y[2] * y[1]

        # MG
        self.dydt[3] = self.k[1] * y[2] * y[1] - self.k[2] * y[3] * y[1]

        # GL
        self.dydt[4] = self.k[2] * y[3] * y[1]

        # R'CO2R
        self.dydt[5] = (self.k[0] * y[0] * y[1] +
                        self.k[1] * y[2] * y[1] + self.k[2] * y[3] * y[1])

        return self.dydt


def get_solution(ode, y0, t_end, n_steps):
    '''Use solve_ivp from scipy to solve the ODE'''
    sol = solve_ivp(ode,
                    t_span=[0, t_end],
                    y0=y0,
                    t_eval=np.linspace(0, t_end, n_steps),
                    method='BDF',
                    dense_output=False,
                    vectorized=False,
                    rtol=1e-3,
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

        self.w_in = nn.Parameter(torch.zeros(ns+1, nr))
        self.w_b = nn.Parameter(torch.zeros(1, nr))
        self.w_out = nn.Parameter(torch.zeros(nr, ns))

        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

        # adaptive weights
        self.slope = nn.Parameter(torch.Tensor([1.0]))
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
        w_in = self.w_in.abs().T.data.numpy() * nslope
        w_in[:, -1] *= Ea_scale

        print(w_in)

        print('w_b')

        w_out_max = self.w_out.abs().max(dim=1).values

        scaled_k = torch.exp(self.w_b*nslope) * w_out_max

        print(scaled_k.detach().numpy())

        print('w_out')

        scaled_w_out = self.w_out.T / w_out_max

        print(scaled_w_out.detach().numpy().T)

    def share_params(self):
        self.w_in.data[:-1, :] = (-self.w_out.T.data).clamp(0)
        # for i in range(self.ns):
        #     for j in range(self.ns):
        #         if self.w_in.data[i, j].item() > 0.8:
        #             self.w_out.data[j, i] = - self.w_in.data[i, j]

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
    def __init__(self, crnn_model, T=300):

        self.R = 1.98720425864083E-3

        self.crnn_model = crnn_model

        self.dydt = []

        self.T = torch.Tensor([-Ea_scale / self.R / T]).view(1, 1)

    def __call__(self, t, y):

        with torch.no_grad():
            y_log = torch.log(torch.Tensor(y).clamp(1e-6)).view(-1, 6)
            input = torch.cat((y_log, self.T), dim=-1)
            dydt = self.crnn_model(input)[0]

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
    for i in range(6):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(label[:, i], pred[:, i].data.numpy(), 'o')
        ax.set_xlabel('Label')
        ax.set_ylabel('Pred')
    ax = fig.add_subplot(2, 4, 6+1)
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
    T_exp_list = np.array([50, 55, 60, 65]) + 273.15  # C -> K
    k_list = np.zeros((3, 4))
    k_list[0] = np.array([0.018, 0.024, 0.036, 0.048])
    k_list[1] = np.array([0.036, 0.051, 0.070, 0.098])
    k_list[2] = np.array([0.112, 0.158, 0.141, 0.191])

    R = 1.98720425864083E-3
    A_list = np.zeros(3)
    Ea_list = np.zeros(3)

    for i in range(3):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=1/T_exp_list, y=np.log(k_list[i]))  # noqa E501
        print("Reaction {}: A = {:.2e} Ea = {:.2f} kcal/mol r_value = {:.4f} p_value = {:.4f}".
              format(i, np.exp(intercept), R*slope, r_value, p_value))
        A_list[i] = np.exp(intercept)
        Ea_list[i] = R*slope

    np.savez('./log/Arrhenius_Paras.npz', A_list=A_list, Ea_list=Ea_list)

    y_list = []
    dydt_list = []
    T_list = np.random.uniform(50, 80, args.n_exp) + 273

    for i in range(args.n_exp):
        k_vals = np.zeros(3)
        for ir in range(3):
            k_vals[ir] = A_list[ir]*np.exp(Ea_list[ir]/R/T_list[i])
        y0 = np.random.uniform(np.array([0, 0, 0, 0, 0, 0]),
                               np.array([1, 1, 1, 1, 1, 0]))
        ode = ReactorOde(k=k_vals, ns=6)
        y, dydt = get_solution(ode, y0, args.t_end, args.n_steps)
        y_T = np.zeros((y.shape[0]+1, y.shape[1]))
        y_T[:-1, :] = y
        y_T[-1, :] = np.exp(-Ea_scale/R/T_list[i])
        y_list.append(y_T.T)
        dydt_list.append(dydt.T)
    y_np = np.vstack(y_list)
    eps = 1e-30
    y_np = np.clip(y_np, eps, None)
    dydt_np = np.vstack(dydt_list)

    # Train CRNN Model

    crnn_model = CRNN_Model(ns=args.ns, nr=args.nr)
    optimizer = torch.optim.Adam(crnn_model.parameters(),
                                 lr=args.learning_rate)
    loss_func = torch.nn.MSELoss()

    X_train, X_test, Y_train, Y_test = train_test_split(y_np,
                                                        dydt_np,
                                                        test_size=0.33,
                                                        random_state=32)

    Y_max = torch.Tensor(dydt_np).abs().max(dim=0).values

    train_data = ODEDataSet(data=np.log(X_train), label=Y_train)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=False)

    test_data = ODEDataSet(data=np.log(X_test), label=Y_test)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=False)

    loss_list = {'epoch': [], 'train': [], 'test': [], 'nslope': []}

    epoch_old = 0
    if args.is_restart:
        checkpoint = torch.load(checkfile + '.tar')
        crnn_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_old = checkpoint['epoch']
        loss_list = checkpoint['loss_list']

        if args.is_pruning:
            for p in crnn_model.parameters():
                mask = p.data.abs() < args.pruning_threshhold
                p.data[mask] *= 0

    alpha = 0e-6

    for epoch in tqdm(range(args.nepochs)):
        if args.is_restart:
            if epoch < epoch_old:
                continue
        loss_train = 0
        i_sample = 0
        for i_sample, (data, label) in enumerate(train_loader):
            pred = crnn_model(data)
            loss = loss_func(pred/Y_max, label/Y_max)

            # scaled_w_in = crnn_model.w_in.abs().T * crnn_model.nslope()
            # w_out_max = crnn_model.w_out.abs().max(dim=1).values + eps
            # scaled_k = torch.exp(crnn_model.w_b*crnn_model.nslope()) * w_out_max  # noqa E501
            # scaled_w_out = crnn_model.w_out.T / w_out_max
            # loss_reg = alpha * (scaled_w_in.norm(1) + scaled_w_out.norm(1))
            # loss += loss_reg

            optimizer.zero_grad()
            loss.backward()

            if not args.adaptive_slope:
                crnn_model.slope.grad.data *= 0

            # TODO: use scaled w_in and w_out.
            if args.is_pruning and args.is_restart:
                for p in crnn_model.parameters():
                    mask = p.abs() < 0.01
                    p.grad.data[mask] *= 0
            optimizer.step()
            # crnn_model.share_params()
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

        if epoch % args.print_freq == 0 or epoch == args.nepochs-1:
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

    i = 10
    k_vals = np.zeros(3)
    for ir in range(3):
        k_vals[ir] = A_list[ir]*np.exp(Ea_list[ir]/R/T_list[i])
    ode = ReactorOde(k=k_vals, ns=6)

    # Posterior Validation by Coupling the CRNN into ODE integration
    y0 = np.array([1, 1, 0, 0, 0, 0])

    # here we test the CRNN for am unseen initial condition
    sol = solve_ivp(ode,
                    t_span=[0, args.t_end*10],
                    y0=y0,
                    t_eval=np.linspace(0, args.t_end*10, args.n_steps*10),
                    method='BDF',
                    dense_output=False,
                    vectorized=False)

    odeNN = ReactorOdeNN(crnn_model, T=T_list[i])

    solNN = solve_ivp(odeNN,
                      t_span=[0, args.t_end*10],
                      y0=y0,
                      t_eval=np.linspace(0, args.t_end*10, args.n_steps*10),
                      method='BDF',
                      dense_output=False,
                      vectorized=False)

    fig = plt.figure(figsize=(12, 7))
    for i in range(6):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(sol.t, sol.y[i, :], color='r', ls='solid', label='label')
        ax.plot(solNN.t, solNN.y[i, :], color='b', ls='dashed', label='crnn')
        ax.set_xlabel('Time')
        ax.set_ylabel('Conc.')
        ax.set_title('Species '+str(i+1))
        ax.set_xlim([0, args.t_end])
        ax.legend()
    fig.tight_layout()
    plt.savefig('./fig/ode_crnn', dpi=120)
