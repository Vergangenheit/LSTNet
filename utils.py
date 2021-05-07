import torch
import numpy as np
from numpy import ndarray
from torch.autograd import Variable
from argparse import Namespace
from torch import device, Tensor
from typing import Generator, List, Dict, Union
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
import matplotlib.pyplot as plt


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name: str, train: float, valid: float, device: device, args: Namespace):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalized(args.normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        fin.close()

    def _normalized(self, normalize: int):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train: int, valid: int, test: int):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set: range, horizon: int):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m), device=self.device)
        Y = torch.zeros((n, self.m), device=self.device)

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.dat[idx_set[i], :], device=self.device)

        return [X, Y]

    def get_batches(self, inputs: Tensor, targets: Tensor, batch_size: int, shuffle: bool = True) -> Generator:
        length = len(inputs)
        if shuffle:
            index: Tensor = torch.randperm(length, device=self.device)
        else:
            index: Tensor = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size


class Data_utility_df(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name: str, train: float, valid: float, device: device, args: Namespace):
        self.device = device
        self.P = args.window
        self.h = args.horizon
        self.rawdat: DataFrame = pd.read_csv(file_name, index_col=0)
        # self.rawdat: ndarray = np.loadtxt(fin, delimiter=',')
        print(f"Dataset size is {self.rawdat.shape}")
        self.dat: ndarray = np.zeros(self.rawdat.shape)
        print(f"Data array shape is {self.dat.shape}")
        self.n, self.m = self.dat.shape
        self.scale: ndarray = np.ones(self.m)
        print(f"Scale array size is {self.scale.shape}")
        self._normalized(args.normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale: Tensor = torch.as_tensor(self.scale, device=device, dtype=torch.float)

        tmp: Tensor = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        # fin.close()

    def _normalized(self, normalize: int):
        # normalized by the maximum value of entire matrix.

        if normalize == 0:
            self.dat = self.rawdat

        if normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat.iloc[:, i]))
                self.dat[:, i] = self.rawdat.iloc[:, i] / np.max(np.abs(self.rawdat.iloc[:, i]))

    def _split(self, train: int, valid: int, test: int):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train: List[Tensor] = self._batchify(train_set, self.h)
        self.valid: List[Tensor] = self._batchify(valid_set, self.h)
        self.test: List[Tensor] = self._batchify(test_set, self.h)
        self.testset_idx = self.rawdat.index[valid:self.n]

    def _batchify(self, idx_set: range, horizon: int) -> List[Tensor]:

        n: int = len(idx_set)
        X: Tensor = torch.zeros((n, self.P, self.m), device=self.device)
        Y = torch.zeros((n, self.m), device=self.device)

        for i in range(n):
            end: int = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.as_tensor(self.dat[start:end, :], device=self.device)
            Y[i, :] = torch.as_tensor(self.dat[idx_set[i], :], device=self.device)

        return [X, Y]

    def get_batches(self, inputs: Tensor, targets: Tensor, batch_size, shuffle=True) -> Generator:
        length: int = len(inputs)
        if shuffle:
            index: Tensor = torch.randperm(length, device=self.device)
        else:
            index: Tensor = torch.as_tensor(range(length), device=self.device, dtype=torch.long)
        start_idx = 0
        while start_idx < length:
            end_idx: int = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]

            yield Variable(X), Variable(Y)
            start_idx += batch_size


class MapePlotter:

    def __init__(self, df: DataFrame, mapes_dict: Dict[str, int]):
        """model(str): pred_horizon hours e.g.('1')
        path(str): prediction files path"""
        self.df = df
        self.mapes_dict: Dict = mapes_dict

    def rolling_mape(self, hours_mape: int) -> DataFrame:
        print(self.df.shape)
        self.df['abs(Pred-true)']: Series = np.abs(self.df['predictions'] - self.df['true'])
        d: List = []

        for i in range(0, self.df.shape[0] - hours_mape):
            a: int = sum(self.df['abs(Pred-true)'][i:i + hours_mape])
            b: int = sum(self.df['true'][i:i + hours_mape])
            c: float = 100 * a / b
            d.append(c)

        # prendere la data del inizio di intervallo
        p: List = []
        for i in range(0, self.df.shape[0] - hours_mape):
            f: Union[str, datetime] = self.df['time'].iloc[i]
            p.append(f)
        assert len(p) == len(d)

        df_mape: DataFrame = DataFrame(data={'time': p, 'mape': d})

        return df_mape

    def plotter(self, df: DataFrame, frequency: str):

        #### Parte che calcola i dati con un certo limite di mape - qualità dei dati###
        mean_mape: float = df['mape'].mean()
        size: int = df.shape[0]
        print(size)
        mean: List[float] = [mean_mape] * size
        # Plottare i dati
        fig, ax = plt.subplots(figsize=(20, 10))

        # inserire: p:lista con le date (non prendere dal dataframe non lo plotta) d:mape in %
        ax.plot(df['time'], df['mape'], label=f'mape {frequency}')

        # inserire la linea "limite" con il mape medio dalla riga 46
        ax.plot(df['time'], mean, linestyle='dashed', linewidth=1,
                label='average mape ' + str(round(mean_mape, 2)) + '%')
        ax.legend(fontsize='xx-large')

        # sistemare x asse dove li diciamo che parte dal prima data e ha la lughezza dell vettore p; 48 sta per due
        # giorni intervalli più piccoli non si leggevano - voi potete provare l'intervallo più picolo il grafico non
        # cambia solo i valori sulla x ha:sta per allineamento delle date
        plt.xticks(np.arange(0, df.shape[0], 48), rotation=20, ha='right')
        ax.set_xlabel('date', fontsize=20)
        ax.set_ylabel(f'rolling {frequency} mape', fontsize=20)
        ax.tick_params(axis='x', which='minor', labelsize=1)
        plt.show()

    def plot(self):
        for mape in self.mapes_dict:
            value: int = self.mapes_dict.get(mape)
            df_mape: DataFrame = self.rolling_mape(value)
            self.plotter(df_mape, mape)


