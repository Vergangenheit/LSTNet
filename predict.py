from utils import Data_utility_df, MapePlotter
from argparse import Namespace, ArgumentParser
from torch import device
from torch.nn import Module
from models.LSTNet import Model
from torch.nn.parallel.data_parallel import DataParallel
import argparse
import torch
from typing import Union, Dict
from torch import Tensor, device
from numpy import ndarray
from pandas import DataFrame


def predict(args: Namespace, device: device) -> DataFrame:
    # load data
    Data = Data_utility_df(args.data, 0.6, 0.2, device, args)
    # Load trained model
    with open(args.save, 'rb') as f:
        model: Union[Module, DataParallel] = torch.load(f)
    model.eval()
    # execute model forward pass
    predict: Tensor = None
    test: Tensor = None
    for x, y in Data.get_batches(Data.test[0], Data.test[1], args.batch_size, False):
        output: Tensor = model(x)
        scale: Tensor = Data.scale.expand(output.size(0), Data.m)
        output = output * scale
        y = y * scale
        if predict is None:
            predict: Tensor = output.clone().detach()
            test: Tensor = y
        else:
            predict: Tensor = torch.cat((predict, output.clone().detach()))
            test: Tensor = torch.cat((test, y))

    # scale: Tensor = Data.scale.expand(output.size(0), Data.m)
    # print(scale.size())
    pred_energies = predict[:, 2]
    true_energies = test[:, 2]
    print(predict.size())
    print(test.size())
    print(pred_energies.size())
    print(true_energies.size())
    print(pred_energies.cpu().numpy()[:10])
    print(true_energies.cpu().numpy()[:10])  # corresponds to the first 10 rows of testset
    pred_energies: ndarray = pred_energies.cpu().numpy()
    true_energies: ndarray = true_energies.cpu().numpy()

    df_preds: DataFrame = DataFrame(data={'true': true_energies, 'predictions': pred_energies})
    df_preds['time'] = Data.testset_idx

    return df_preds


if __name__ == "__main__":
    parser: ArgumentParser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--model', type=str, default='LSTNet', help='')
    parser.add_argument('--window', type=int, default=24 * 7, help='window size')
    parser.add_argument('--horizon', type=int, default=12)

    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units each layer')
    parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units (channels)')
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')

    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--amsgrad', type=str, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--skip', type=float, default=24)
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    args: Namespace = parser.parse_args()
    # choose device
    device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_preds: DataFrame = predict(args, device)
    mapes_dict: Dict = {'weekly': 24 * 7, 'bi_weekly': 24 * 14, '700': 700}
    plotter = MapePlotter(df_preds, mapes_dict)
    plotter.plot()

