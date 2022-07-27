import argparse
import pickle
import time
import math
import torch
import sys
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import sparse_rnn_core, data
from sparse_rnn_core import Masking, CosineDecay
from models import RHN, Stacked_LSTM
from Sparse_ASGD import Sparse_ASGD
import numpy as np
from LE_calculation import *
# from train_PBT_selfishRNN import *
from train import train_main
from args import Args
from LE_calculation import *
from util import *

def main():
    args = Args().args
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #
    # args.device = torch.device("cuda" if args.cuda else "cpu")

    # print(args)

    sparse_inits = ['uniform', 'ER']
    growths = ['random']#, 'momentum', 'momentum_neuron']
    deaths = ['magnitude', 'SET', 'threshold', 'global_magnitude']
    redistributions = ['magnitude', 'nonzeros', 'none']#, 'momentum']

    args.epochs = 3
    num_trials = 40
    count = 5024
    trials = {}
    for i in range(num_trials):
        init = sparse_inits[np.random.randint(0, len(sparse_inits))]
        growth = growths[np.random.randint(0, len(growths))]
        death = deaths[np.random.randint(0, len(deaths))]
        redistribution = redistributions[np.random.randint(0, len(redistributions))]
        death_rate = 0.1 * (np.random.randint(7) + 2)  # [0.2, 0.8]
        lr = 10 * (np.random.randint(4) + 1)

        trials[count] = {'init': init, 'growth': growth, 'death': death,
                         'redistribution': redistribution, 'death_rate': death_rate,
                         'lr': lr}

        args.save = f'{count}.pt'
        args.init = init
        args.growth = growth
        args.death = death
        args.redistribution = redistribution
        args.death_rate = death_rate
        args.lr = lr
        print(f"--------------------trial: {i+1} / {num_trials}-------------------")
        print(f'trial index: {count}\n'
              f'init: {init}\t grow: {growth}\n'
              f'death:{death} \t redistribution: {redistribution}\n'
              f'death_rate: {death_rate} \t lr: {lr}')
        args.eval_batch_size = 20
        val_loss = train_main(args, 'cuda')
        print(val_loss)
        args.trial_num = count
        args.eval_batch_size = 2
        LE_main(args)
        count += 1
        print(f"---------------------------------------------\n")
    pickle.dump(trials, open('../trials/LE_trials_num_2_ind_0.pickle', 'wb'))

def continue_competition():
    starting_ind = 18000
    max_evals = 40
    LE_based = True
    remaining_indices = np.array(range(starting_ind, starting_ind + max_evals))
    a = len(remaining_indices)
    if LE_based:
        trials = pickle.load(open(f'../trials/LE_tpe_trials_num_{max_evals}_ind_{starting_ind}.pickle', 'rb'))
    else:
        trials = pickle.load(open(f'../trials/PPL_tpe_trials_num_{max_evals}_ind_{starting_ind}.pickle', 'rb'))
    distances = []
    for trial in trials.trials:
        distances.append(trial['result']['loss'])

    print(remaining_indices)
    print(distances)
    sorted_indices = np.argsort(distances)
    remaining_indices = remaining_indices[sorted_indices[:int(a/2)]]
    print(remaining_indices)

    LE_distances = {}
    val_losses = {}
    test_losses = {}
    remaining_indices = [20, 22]
    remaining_indices = [x + starting_ind for x in remaining_indices]
    epoch = 100
    # epoch = 6
    for i, remaining_index in enumerate(remaining_indices):
        # if i == 0:
        trial = trials.trials[remaining_index - starting_ind]
        args = trial['result']['args']
        args.data = '/home/ws8/caleb/dataset/PTB/penn/'
        args.seed = 233
        print(args)
        args.epochs = epoch
        args.eval_batch_size = 20
        val_loss, test_loss = train_main(args, 'cuda')
        # print(f"count: {remaining_index}, val_loss: {math.exp(val_loss)}")
        val_losses[remaining_index] = round(math.exp(val_loss), 2)
        test_losses[remaining_index] = round(math.exp(test_loss), 2)
        if epoch < 50:
            args.trial_num = remaining_index
            args.eval_batch_size = 2
            LE_main(args)
            LE_distance = LE_distance_main(remaining_index, num_epochs=epoch)
            LE_distances[remaining_index] = round(LE_distance, 2)
        else:
            LE_distance = 0
            LE_distances[remaining_index] = LE_distance


        print(f"count: {remaining_index}, val_loss: {math.exp(val_loss)}, test_loss: {math.exp(test_loss)}, LE_distance: {LE_distance}")
    print("val_losses: ", val_losses)
    print("LE_distances: ", LE_distances)
if __name__ == '__main__':
    # main()
    continue_competition()
