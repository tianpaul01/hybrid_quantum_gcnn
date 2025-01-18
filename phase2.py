from __future__ import print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import csv
import datetime
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

from scipy.io import loadmat

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import time
from datetime import datetime


torch.manual_seed(123)
torch.set_printoptions(precision=20)
dev1 = torch.device("cuda:0")  # First GPU device
dev2 = torch.device("cuda:1")  # Second GPU device
#dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev1, dev2)

def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    
    X = np.array([timeseries[start:start + window_size] for start in range(timeseries.shape[0] - window_size + 1)])
    X = np.transpose(X, (0, 2, 1))
    
    return X

def read_file(fn):
    vals = []
    with open(fn,'r') as csvfile:
            tsdata=csv.reader(csvfile, delimiter=',')
            for row in tsdata:
                    vals.append(row)
    vals = vals[1:]
    y = np.array(vals).astype(float)
    return y
    
locA = np.array(read_file(r'./data/summer/a.csv')).astype(float)
locB = np.array(read_file(r'./data/summer/b.csv')).astype(float)
locC = np.array(read_file(r'./data/summer/c.csv')).astype(float)
locD = np.array(read_file(r'./data/summer/d.csv')).astype(float)
locE = np.array(read_file(r'./data/summer/e.csv')).astype(float)
locF = np.array(read_file(r'./data/summer/f.csv')).astype(float)
locG = np.array(read_file(r'./data/summer/g.csv')).astype(float)
locH = np.array(read_file(r'./data/summer/h.csv')).astype(float)

def combination(loc):
    location_mapping = {
        0: (locA, locB, locC),
        1: (locA, locB, locD),
        2: (locA, locB, locE),
        3: (locA, locB, locF),
        4: (locA, locB, locG),
        5: (locA, locB, locH),
        6: (locA, locC, locD),
        7: (locA, locC, locE),
        8: (locA, locC, locF),
        9: (locA, locC, locG),
        10: (locA, locC, locH),
        11: (locA, locD, locE),
        12: (locA, locD, locF),
        13: (locA, locD, locG),
        14: (locA, locD, locH),
        15: (locA, locE, locF),
        16: (locA, locE, locG),
        17: (locA, locE, locH),
        18: (locA, locF, locG),
        19: (locA, locF, locH),
        20: (locA, locG, locH),
        21: (locA, locB, locC, locD),
        22: (locA, locB, locC, locE),
        23: (locA, locB, locC, locF),
        24: (locA, locB, locC, locG),
        25: (locA, locB, locC, locH),
        26: (locA, locB, locD, locE),
        27: (locA, locB, locD, locF),
        28: (locA, locB, locD, locG),
        29: (locA, locB, locD, locH),
        30: (locA, locB, locE, locF),
        31: (locA, locB, locE, locG),
        32: (locA, locB, locE, locH),
        33: (locA, locB, locF, locG),
        34: (locA, locB, locF, locH),
        35: (locA, locC, locD, locE),
        36: (locA, locC, locD, locF),
        37: (locA, locC, locD, locG),
        38: (locA, locC, locD, locH),
        39: (locA, locC, locE, locF),
        40: (locA, locC, locE, locG),
        41: (locA, locC, locE, locH),
        42: (locA, locC, locF, locG),
        43: (locA, locC, locF, locH),
        44: (locA, locD, locE, locF),
        45: (locA, locD, locE, locG),
        46: (locA, locD, locE, locH),
        47: (locA, locE, locF, locG),
        48: (locA, locE, locF, locH),
        49: (locA, locB, locC, locD, locE),
        50: (locA, locB, locC, locD, locF),
        51: (locA, locB, locC, locD, locG),
        52: (locA, locB, locC, locD, locH),
        53: (locA, locB, locC, locE, locF),
        54: (locA, locB, locC, locE, locF),
        55: (locA, locB, locC, locE, locG),
        56: (locA, locB, locC, locE, locH),
        57: (locA, locB, locC, locF, locG),
        58: (locA, locB, locC, locF, locH),
        59: (locA, locB, locD, locE, locF),
        60: (locA, locB, locD, locE, locG),
        61: (locA, locB, locD, locE, locH),
        62: (locA, locB, locE, locF, locG),
        63: (locA, locC, locD, locE, locF),
        64: (locA, locC, locD, locE, locG),
        65: (locA, locC, locD, locE, locH),
        66: (locA, locC, locE, locF, locG),
        67: (locA, locC, locE, locF, locH),
        68: (locA, locD, locE, locF, locG),
        69: (locA, locD, locE, locF, locH),
        70: (locA, locB, locC, locD, locE, locF),
        71: (locA, locB, locC, locD, locE, locG),
        72: (locA, locB, locC, locD, locE, locH),
        73: (locA, locB, locC, locE, locF, locG),
        74: (locA, locB, locC, locE, locF, locH),
        75: (locA, locB, locD, locE, locF, locG),
        76: (locA, locB, locD, locE, locF, locH),
        77: (locA, locC, locD, locE, locF, locG),
        78: (locA, locC, locD, locE, locF, locH),
        79: (locA, locB, locC, locD, locE, locF, locG),
        80: (locA, locB, locC, locD, locE, locF, locH),
        81: (locA, locB, locC, locD, locE, locF, locG, locH)
    }
    
    combined_data = np.concatenate(location_mapping[loc], axis=1)

    # Calculate minimum and maximum values
    min_value = np.min(combined_data)
    max_value = np.max(combined_data)

    # Normalize data to (0, 1) range
    normalized_data = (combined_data - min_value) / (max_value - min_value)

    return normalized_data

class DatasetPrep():
    def __init__(self, inputTimesteps,train,training,test, predictTimestep=24):
        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep

        if train:
            x = training
        else:
            x = test

        # original order: T, V, C (vertex/city, timestep, channel/variable)
        # now the order is changed into: C, T, V
        self.x = torch.tensor(x).permute(2, 0, 1).double() 


    def __getitem__(self, item):
        x = self.x[:, item:item + self.inputTimesteps, :]
        y = self.x[0, item + self.inputTimesteps + self.predictTimestep - 1, 0] # WIND SPEED of all the 7 cities (feature no. 0)

        return x, y

    def __len__(self):
        return self.x.shape[1] - self.inputTimesteps - self.predictTimestep + 1


def get_train_valid_loader(input_timesteps,training,test):
    train_val_dataset = DatasetPrep(inputTimesteps=input_timesteps, train=True, training=training, test=test)
    num_train = len(train_val_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(123)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_val_dataset, batch_size=2000, sampler=train_sampler)
    valid_loader = DataLoader(train_val_dataset, batch_size=2000, sampler=valid_sampler)

    return train_loader, valid_loader


def get_test_loader(input_timesteps):
    test_dataset = DatasetPrep(inputTimesteps=input_timesteps, train=False, training=training, test=test)

    data_loader = DataLoader(test_dataset, batch_size=2000)

    return data_loader
    
import torch.nn as nn
import torch
import math
from torch.autograd import Variable


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_vertices):
        super(unit_gcn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.B = nn.Parameter(torch.zeros(num_vertices, num_vertices) + 1e-6)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.A = Variable(torch.eye(num_vertices), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(1e-6))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)

    def forward(self, x):
        N, C, T, V = x.size()

        f_in = x.contiguous().view(N, C * T, V)

        adj_mat = None
        self.A = self.A.cuda(x.get_device())
        adj_mat = self.B[:V, :V] + self.gamma * self.A[:V, :V]
        adj_mat_min = torch.min(adj_mat)
        adj_mat_max = torch.max(adj_mat)
        adj_mat = (adj_mat - adj_mat_min) / (adj_mat_max - adj_mat_min)

        D = Variable(torch.diag(torch.sum(adj_mat, axis=1)), requires_grad=False)
        D_12 = torch.sqrt(torch.inverse(D))
        adj_mat_norm_d12 = torch.sparse.mm(torch.sparse.mm(D_12, adj_mat), D_12).to_dense()  # Convert sparse matrix to dense

        # Reshape adj_mat_norm_d12 to match the shape of f_in for multiplication
        adj_mat_norm_d12 = adj_mat_norm_d12.unsqueeze(0).expand(N, -1, -1)

        y = self.conv(torch.bmm(f_in, adj_mat_norm_d12).view(N, C, T, V))  # Use torch.bmm for batch matrix multiplication

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_vertices, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, num_vertices)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.gcn1(x) + self.residual(x)
        x = self.tcn1(x)
        x = self.relu(x)

        return x

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np  # Don't forget to import numpy

class Model(nn.Module):
    def __init__(self,qembed=0, num_qubits=10, n_layers=6, num_layers=3, num_vertices=7, channel1=16, channel2=32, channel3=64, conv_dim=4, timesteps=30):
        super(Model, self).__init__()
        
        dev = qml.device("default.qubit", wires=num_qubits)
        @qml.qnode(dev,diff_method="best", interface="torch")
        def iqp_embed(inputs, weights):
            qml.IQPEmbedding(inputs, wires=range(num_qubits))  # Use IQPEmbedding
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        @qml.qnode(dev,diff_method="best", interface="torch")
        def angle_embed(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits))  # Use Angle Embedding
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        @qml.qnode(dev,diff_method="best", interface="torch")
        def ampl_embed(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(num_qubits))  # Use Amplitude Embedding
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        

        weight_shapes = {"weights": (n_layers, num_qubits, 3)}
        init_method = {"weights": torch.normal(0,0.1/math.sqrt(num_layers), size=(n_layers,num_qubits,3))}
        self.data_bn = nn.BatchNorm1d((num_vertices - 1) * num_vertices)
        
        if num_layers == 1:
            self.layer1 = TCN_GCN_unit((num_vertices - 1), channel1, num_vertices)
            self.layer2 = None
            self.layer3 = None
            self.conv_reduce_dim = unit_tcn(channel1, conv_dim, kernel_size=1, stride=1)
        elif num_layers == 2:
            self.layer1 = TCN_GCN_unit((num_vertices - 1), channel1, num_vertices)
            self.layer2 = TCN_GCN_unit(channel1, channel2, num_vertices)
            self.layer3 = None
            self.conv_reduce_dim = unit_tcn(channel2, conv_dim, kernel_size=1, stride=1)
        elif num_layers == 3:
            self.layer1 = TCN_GCN_unit((num_vertices - 1), channel1, num_vertices)
            self.layer2 = TCN_GCN_unit(channel1, channel2, num_vertices)
            self.layer3 = TCN_GCN_unit(channel2, channel3, num_vertices)
            self.conv_reduce_dim = unit_tcn(channel3, conv_dim, kernel_size=1, stride=1)
        dim = timesteps * conv_dim * num_vertices
        self.fc = nn.Linear(dim, num_qubits)
        self.fc2 = nn.Linear(num_qubits, 1)
        self.fc3 = nn.Linear(1, 24)  # New fully connected layer
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_vertices))
        bn_init(self.data_bn, 1)
        
        if qembed == 0:
            self.qlayer = qml.qnn.TorchLayer(qnode=iqp_embed, weight_shapes=weight_shapes, init_method=init_method)
        else:
            self.qlayer = qml.qnn.TorchLayer(qnode=angle_embed, weight_shapes=weight_shapes, init_method=init_method)
        
        self.biasq = nn.Parameter(torch.zeros(num_qubits))
        
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)
        x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        x = self.conv_reduce_dim(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.qlayer(x)
        x = x + self.biasq
        x = self.fc2(x)
        x = self.fc3(x)  # Apply the new fully connected layer
        
        return x
        
import pytorch_lightning as pl
class QuantumModel(pl.LightningModule):
    def __init__(self, qembed=0, num_qubits=10, n_layers=6, num_layers=3, num_vertices=7, channel1=16, channel2=32, channel3=64, conv_dim=4, timesteps=30):
        super(QuantumModel, self).__init__()
        self.model = Model(qembed,num_qubits, n_layers, num_layers, num_vertices, channel1, channel2, channel3, conv_dim, timesteps)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.MSELoss()(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        
from mealpy.optimizer import Optimizer
from mealpy.utils.agent import Agent
import numpy as np

class ChaosPPSO(Optimizer):
    """
    The CPPSO-CD algorithm integrates chaotic dynamics into the standard PPSO framework to enhance its exploration and exploitation capabilities.
    """

    def __init__(self, epoch: int = 10000, pop_size: int = 100, chaos_factor: float = 0.5, cauchy_scale: float = 1.0, **kwargs: object) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            chaos_factor: factor to introduce chaotic behavior, default = 0.5
            cauchy_scale: scale parameter for Cauchy distribution, default = 1.0
        """
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.chaos_factor = chaos_factor
        self.cauchy_scale = cauchy_scale
        self.set_parameters(["epoch", "pop_size", "chaos_factor", "cauchy_scale"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.dyn_delta_list = self.generator.uniform(0, 2 * np.pi, self.pop_size)

    def generate_empty_agent(self, solution: np.ndarray = None) -> Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.v_max, self.v_max)
        local_pos = solution.copy()
        return Agent(solution=solution, velocity=velocity, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray = None) -> Agent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of CPPSO-CD algorithm with chaos and Cauchy mechanisms.
        """
        for idx in range(0, self.pop_size):
            # Calculate phasor coefficients
            aa = 2 * (np.sin(self.dyn_delta_list[idx]))
            bb = 2 * (np.cos(self.dyn_delta_list[idx]))
            ee = np.abs(np.cos(self.dyn_delta_list[idx])) ** aa
            tt = np.abs(np.sin(self.dyn_delta_list[idx])) ** bb

            # Velocity update with chaotic dynamics and Cauchy noise
            v_new = ee * (self.pop[idx].local_solution - self.pop[idx].solution) + \
                    tt * (self.g_best.solution - self.pop[idx].solution)
            cauchy_noise = np.random.standard_cauchy(size=self.problem.n_dims) * self.cauchy_scale
            v_new = v_new + cauchy_noise
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            self.pop[idx].velocity = v_new

            # Position update
            pos_new = self.pop[idx].solution + v_new
            pos_new = self.correct_solution(pos_new)

            # Phase angle and max velocity updates
            self.dyn_delta_list[idx] += np.abs(np.cos(self.dyn_delta_list[idx]) + np.sin(self.dyn_delta_list[idx])) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[idx])) ** 2) * (self.problem.ub - self.problem.lb)

            # Evaluate target and update best solutions
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(target, self.pop[idx].local_target, self.problem.minmax):
                self.pop[idx].update(local_solution=pos_new.copy(), local_target=target.copy())

from torch.utils.tensorboard import SummaryWriter
from torch import amp 
from torchmetrics import R2Score
import torch
import datetime
import pytorch_lightning as pl
import logging
import torch.nn as nn


def fitness_function(x):
    t1_0 = datetime.datetime.now()
    now = datetime.datetime.now()
    date_time_string = f"{now.year}{str(now.month).zfill(2)}{str(now.day).zfill(2)}_{str(now.hour + 2).zfill(2)}{str(now.minute).zfill(2)}{str(now.second).zfill(2)}"

    loc = #loc variable
    input_timesteps = #timestep
    num_layers = #layers
    a =  #channel1
    b =  #channel2
    c =  #channel3
    conv_dim =  #conv. red. 
    num_qubits = int(x[0])
    n_layers = int(x[1])
    qembed = int(x[2])
    epochs = 500

    combinations = combination(loc)
    vertices = combinations.shape[1]
    print(f"timestep: {input_timesteps}, loc: {loc}, vert: {vertices}, layers: {num_layers}, chan1: {a}, "
          f"chan2: {b}, chan3: {c}, conv: {conv_dim}, embed: {qembed}, qubits: {num_qubits}, qlayer: {n_layers}")

    channels = vertices - 1
    X = make_timeseries_instances(combinations, channels)
    num = int(X.shape[0] * 0.8)
    training, test = X[:num, :], X[num:, :]
    train_dl, valid_dl = get_train_valid_loader(input_timesteps, training=training, test=test)

    model = QuantumModel(
        qembed=qembed,
        num_qubits=num_qubits,
        n_layers=n_layers,
        num_layers=num_layers,
        num_vertices=vertices,
        channel1=a,
        channel2=b,
        channel3=c,
        conv_dim=conv_dim,
        timesteps=input_timesteps
    )
    model = model.double().to(dev)
    writer = SummaryWriter()
    loss_func = nn.L1Loss()
    opt = torch.optim.Adam(model.parameters())
    scaler = amp.GradScaler('cuda')
    best_mean_valid_loss = float("inf")

    logging.getLogger("pennylane").setLevel(logging.WARNING)
    pl._logger.handlers[0].setLevel(logging.WARNING)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="mean_valid_loss", patience=5, mode="min")
    trainer = pl.Trainer(callbacks=[early_stop_callback], logger=False)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb[:, :, :input_timesteps].to(dev)
            yb = yb.unsqueeze(1).to(dev)

            with amp.autocast('cuda'):
                pred = model(xb)
                yb = yb.expand_as(pred)
                loss = loss_func(pred, yb)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        valid_loss = 0.0
        valid_rmse = 0.0
        valid_mbe = 0.0
        valid_mase = 0.0
        naive_denom_sum = 0.0
        valid_num = 0

        with torch.no_grad(), amp.autocast('cuda'):
            for xb, yb in valid_dl:
                xb = xb[:, :, :input_timesteps].to(dev)
                yb = yb.unsqueeze(1).to(dev)
                pred = model(xb)

                # Compute metrics
                errors = pred - yb
                abs_errors = torch.abs(errors)
                valid_loss += loss_func(pred, yb).item()
                valid_rmse += torch.sqrt(torch.mean(errors ** 2)).item()
                
    
            # Finalize metrics
            mean_valid_loss = valid_loss / valid_num
            mean_valid_rmse = valid_rmse / valid_num
          

            if mean_valid_loss < best_mean_valid_loss:
                torch.save(model.state_dict(), f"trained_models/cppsocauchy_summer_{date_time_string}.pt")
                best_mean_valid_loss = mean_valid_loss

        writer.add_scalar("Loss/Train", loss.item(), epoch)
        writer.add_scalar("Loss/Valid", best_mean_valid_loss, epoch)

    writer.close()
    torch.cuda.empty_cache()
    t2_0 = datetime.datetime.now()
    time = t2_0 - t1_0

    with open("output.txt", "a") as f:
        output_text = (f"timestep: {input_timesteps}, loc: {loc}, vert: {vertices}, layers: {num_layers}, "
                       f"chan1: {a}, chan2: {b}, chan3: {c}, conv: {conv_dim}, embed: {qembed}, qubits: {num_qubits}, "
                       f"qlayer: {n_layers}\n time: {time}, MAE: {best_mean_valid_loss}, RMSE: {mean_valid_rmse}, "
                       f"file: cppsoq_summer_{date_time_string}\n")
        f.write(output_text)

    print(f"time: {time}, MAE: {best_mean_valid_loss}, RMSE: {mean_valid_rmse}, "
          f"file: cppsoq_summer_{date_time_string}")
    return best_mean_valid_loss

import numpy as np
import datetime
from mealpy import FloatVar
problem_dict1 = {
    "obj_func": fitness_function,
    "bounds": FloatVar(lb = [2,1,0], ub=[8.99,3.99,2.99]),
    "minmax": "min",
    "log_to": "file",
    "log_file": "logs/cppsoq_summer.log",         # Default value = "mealpy.log"
    "name": "GCNN-QNN-summer-CPPSO",
    "max_early_stop": 20,
}
epoch = 50
pop_size = 10
t1_1 = datetime.datetime.now()
modelppso = ChaosPPSO(epoch, pop_size)
best_position, best_fitness = modelppso.solve(problem_dict1) 
print(f"Solution: {best_position}, Fitness: {best_fitness}")
t2_1 = datetime.datetime.now()
print("GPU computation time for ChaosPPSOCauchy: " + str(t2_1-t1_1))



