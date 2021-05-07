import math
import torch.optim as optim
import torch
import numpy as np
from numpy import ndarray
from torch.nn import Module, MSELoss, L1Loss
from torch import Tensor
from utils import Data_utility
from argparse import Namespace
from torch.optim import Adadelta, Adagrad, Adam, SGD
from typing import Union, TypeVar

np.seterr(divide='ignore', invalid='ignore')
torch.backends.cudnn.enabled = False
Optimizer = TypeVar("optimizer", SGD, Adagrad, Adadelta, Adam)


def evaluate(data: Data_utility, X: Tensor, Y: Tensor, model: Module, evaluateL2: MSELoss, evaluateL1: L1Loss,
             args: Namespace) -> (float, float, float):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, args.batch_size, False):
        output: Tensor = model(X)
        if predict is None:
            predict: Tensor = output.clone().detach()
            test = Y
        else:
            predict: Tensor = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += float(evaluateL2(output * scale, Y * scale).data.item())
        total_loss_l1 += float(evaluateL1(output * scale, Y * scale).data.item())

        n_samples += int((output.size(0) * data.m))

    rse: float = math.sqrt(total_loss / n_samples) / data.rse
    rae: float = (total_loss_l1 / n_samples) / data.rae

    predict: ndarray = predict.data.cpu().numpy()
    Ytest: ndarray = test.data.cpu().numpy()
    sigma_p: float = predict.std(axis=0)
    sigma_g: float = Ytest.std(axis=0)
    mean_p: float = predict.mean(axis=0)
    mean_g: float = Ytest.mean(axis=0)
    index: bool = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation: float = (correlation[index]).mean()
    return rse, rae, correlation


def train(data: Data_utility, X: Tensor, Y: Tensor, model: Module, criterion: Union[MSELoss, L1Loss], optim: Optimizer,
          args: Namespace):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, args.batch_size, True):
        optim.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optim.step()
        total_loss += loss.data.item()
        n_samples += int((output.size(0) * data.m))
    return total_loss / n_samples


def makeOptimizer(params, args) -> Union[SGD, Adagrad, Adadelta, Adam]:
    if args.optim == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, )
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr, )
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=args.lr, )
    elif args.optim == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, )
    else:
        raise RuntimeError("Invalid optim method: " + args.method)
    return optimizer
