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


# from train_PBT_selfishRNN import *
from sources.args import Args
from sources.train import train_main

def main():
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
    main()

