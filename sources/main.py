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
from train import *


def main(seed, init, opt, death='magnitude', growth='random', redistribution='magnitude',
         death_rate=0.8, lr=40, clip=0.25, dropout=0.65):
    parser = argparse.ArgumentParser(description='Calculate LE from trained model')
    args = parser.parse_args()
    args.data = '/home/ws8/caleb/dataset/PTB/penn/'  # location of the data corpus
    args.model = 'LSTM'  # type of recurrent net (RHN, LSTM)
    args.evaluate = ''  # path to pre-trained model (default: none)
    args.emsize = 1500  # size of word embeddings
    args.nhid = 1500  # number of hidden units per layer
    args.nonmono = 5  # random seed ?
    args.nlayers = 2  # number of layers
    args.nrecurrence_depth = 10  # number of recurrence layer
    args.momentum = 0.9  # SGD momentum (default: 0.0)
    args.beta = 1  # beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
    args.finetuning = 100  # When (which epochs) to switch to finetuning
    args.lr = 15  # initial learning rate
    args.clip = 0.25  # gradient clipping
    args.epochs = 100  # upper epoch limit
    args.batch_size = 20  # batch size
    args.eval_batch_size = 20  # evaluate batch size
    args.bptt = 35  # sequence length
    args.dropout = 0.65  # dropout applied to layers (0 = no dropout)
    args.dropouth = 0.25  # dropout for rnn hidden units (0 = no dropout)
    args.dropouti = 0.65  # dropout for input embedding layers (0 = no dropout)
    args.dropoute = 0.2  # dropout to remove words from embedding layer (0 = no dropout)
    args.tied = True
    args.couple = True
    args.seed = 1111
    args.cuda = True
    args.log_interval = 200
    args.wdecay = 1.2e-6
    args.optimizer = 'sgd'
    args.testmodel = '15877701169307067.pt'
    randomhash = ''.join(str(time.time()).split('.'))
    args.save = randomhash + '.pt'

    # customized hyperparameters
    args.epochs = 100
    args.couple = False
    args.tied = False
    args.cuda = True
    args.lr = lr
    args.optimizer = opt
    args.seed = seed
    args.save = f'{seed}.pt'
    args.eval_batch_size = 2
    args.clip = clip
    args.dropout = dropout
    # args.evaluate = '../models/___e99___329.pt'
    # args.evaluate = '../../../dataset/PTB/models/stacked_LSTM_pruned/___e99___320.pt'
    print(os.path.exists(args.evaluate))
    # customized sparse hyperparameters
    sparse_rnn_core.add_sparse_args(parser)
    args.sparse = True
    args.fix = False
    args.sparse_init = init
    args.death = death
    args.death_rate = death_rate
    args.density = 0.2
    args.growth = growth
    args.redistribution = redistribution
    args.update_frequency = 100
    args.decay_schedule = 'cosine'
    print(args)

    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    args.device = torch.device("cuda" if args.cuda else "cpu")

    train_main(args, args.device)
    # LE_main(args)


if __name__ == '__main__':
    init = 'uniform'

    seed = 10
    # deaths = ['magnitude', 'SET', 'threshold']
    # growths = ['random']
    # redistributions = ['magnitude', 'nonzeros', 'none']
    deaths = ['threshold']
    growths = ['random']
    redistributions = ['none']
    death_rates = [0.4, 0.8]
    lrs = [20, 40]
    clips = [0.25, 0.5]
    dropouts = [0.2, 0.65]
    optimizers = ['sgd', 'adam']

    # num_exp = 1
    num_exp = len(deaths) * len(growths) * len(redistributions) * len(death_rates) * len(lrs) \
              * len(clips) * len(dropouts) * len(optimizers)
    count = 0
    promising_trials = [356]
    # main(321 + count, init, optimizer)
    for a, death_rate in enumerate(death_rates):
        for b, lr in enumerate(lrs):
            for c, clip in enumerate(clips):
                for d, dropout in enumerate(dropouts):
                    for e, optimizer in enumerate(optimizers):
                        print(f'\n------ {count + 1} / {num_exp} -----')
                        print(f'death rate: {death_rate}, learning rate: {lr}, gradient clip: {clip}, '
                              f'dropout: {dropout}, optimizer: {optimizer}')
                        if (count + 340) in promising_trials:
                            main(340 + count, init, opt=optimizer, death_rate=death_rate, lr=lr,
                                 clip=clip, dropout=dropout)
                        count += 1
                        print(f'--------------------------')
