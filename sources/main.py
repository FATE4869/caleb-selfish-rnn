import argparse
import time
import math
import torch
import sys
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sources import sparse_rnn_core, data
from sources.sparse_rnn_core import Masking, CosineDecay
from sources.models import RHN, Stacked_LSTM
from sources.Sparse_ASGD import Sparse_ASGD
from LE_calculation import *
from util import *
import argparse
import time
import math
import torch
import sys
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import data
# from sources import data
from sparse_rnn_core import Masking, CosineDecay
from models import RHN, Stacked_LSTM
from Sparse_ASGD import Sparse_ASGD
from LE_calculation import *
# from train_PBT_selfishRNN import *
from sources.args import Args
from sources.train import train_main

def trials_train():
    trials = pickle.load(open(f'../trials/tpe_trials_num_20.pickle', 'rb'))

    for i, trial in enumerate(trials):
        if i == 11:
            print(trial)
    #     if i == 1:
            args = trial['result']['args']
            args.epochs = 100
            args.seed = 1111
            args.trial_num = 8003
            args.save='8003.pt'
            args.lr = 40
            # args.eval_batch_size = 20
            print(args)
            val_loss = train_main(args, 'cuda')
            args.eval_batch_size = 2
            LE_main(args)
        # LE_distance = LE_distance_main(trial['result']['args'].trial_num, num_epochs=10,
        #                                epoch=10, last_epoch_ref=True)
        # print(LE_distance)

def model_testing():
    # import os
    # print(os.path.exists(f'../models/stacked_LSTM_pruned/___e99___18020.pt'))
    # filename = f'../models/stacked_LSTM_pruned/___e100___18020.pt'
    # saved = torch.load(filename)
    # print(saved.keys())
    args = Args().args
    args.sparsity = 0.67
    args.density = 1 - args.sparsity
    args.eval_batch_size = 20
    args.seed = 1111

    args.init = 'uniform'
    args.growth = 'random'
    args.death = 'magnitude'
    args.redistribution = 'magnitude'
    # args.death_rate = 0.001 * (params['death_rate'] + 400)
    args.death_rate = 0.1 * (6)
    # args.lr = 10 * (params['lr'] + 1)
    args.verbose = False

    args.evaluate = f'../models/stacked_LSTM_pruned/___e100___18020.pt'
    train_main(args, 'cuda')

if __name__ == '__main__':


    # seed = 10
    # sparse_init = ['uniform', 'ER']
    # deaths = ['magnitude', 'SET', 'threshold', 'global_magnitude']
    # growths = ['random']
    # redistributions = ['magnitude', 'nonzeros', 'none']

    # sparse_inits = ['uniform', 'ER']
    # "sparse_init": hp.choice("sparse_init", ['uniform', 'ER']),
    # "growth": hp.choice("growth", ['random']),
    # "death": hp.choice("death", ['magnitude', 'SET', 'threshold', 'global_magnitude']),
    # "redistribution": hp.choice("redistribution", ['magnitude', 'nonzeros', 'none']),
    # "death_rate": hp.randint('death_rate', 7),  # Returns a random integer in the range [0, upper)
    # "lr": hp.randint('lr', 4)  # Returns a random integer in the range [0, upper)
    model_testing()

