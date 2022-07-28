'''
This code will load trained model and calculate the LEs

gru model are stored in 'trials/gru/models/...'
each .pickle file contains 50 trials and each trial is trained with 20 epochs, the model and accuracy
after each epoch of training is saved. You can just load the model to perform the LEs calculation

lstm model:
'''
import argparse
import time
import math
import pickle
import torch
# from dataloader import MNIST_dataloader
import torch.nn as nn
from models import Stacked_LSTM
from lyapunov import calc_LEs_an
import os
import data
import numpy as np
from args import Args

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(args, model, data_source, corpus, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.eval_batch_size)

    with torch.no_grad():
        start = time.time()
        for i, idx in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
            data, targets = get_batch(data_source, idx, args.bptt)

            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()

            if i < 5:
                h = torch.zeros(args.nlayers, data.size(1), args.nhid).to(args.device)
                c = torch.zeros(args.nlayers, data.size(1), args.nhid).to(args.device)
                emb = model.drop(model.encoder(data))
                params = (emb, (h, c))
                LEs, rvals = calc_LEs_an(*params, model=model, rec_layer='lstm')

                if i == 0:
                    LE_list = LEs
                else:
                    LE_list = torch.cat([LE_list, LEs], dim=0)
                    # print(LEs, rvals)

        LE_maxs = torch.zeros(10)
        LE_means = torch.zeros(10)
        LE_mins = torch.zeros(10)
        LE_vars = torch.zeros(10)
        LEs_avg = torch.zeros(10, 3000)
        for i in range(10):
            if i == 0:
                LEs_avg[i] = LE_list[0]
            else:
                LEs_avg[i] = torch.mean(LE_list[:i], dim=0)
            LE_maxs[i] = torch.max(LEs_avg[i])
            LE_mins[i] = torch.min(LEs_avg[i])
            LE_means[i] = torch.mean(LEs_avg[i])
            LE_vars[i] = torch.var(LEs_avg[i])
        # for j in range(LEs.shape[0]* 2):
        #     LE_maxs[j] = torch.max(LE_list[j, :])
        #     LE_mins[j] = torch.min(LE_list[j, :])
        #     LE_means[j] = torch.mean(LE_list[j, :])
        #     LE_vars[j] = torch.var(LE_list[j, :])
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.scatter(range(10), LE_means, label='mean', c='g')
        # plt.scatter(range(10), LE_maxs, label='max', c='r')
        # plt.scatter(range(10), LE_mins, label='min', c='b')
        # plt.legend()
        # plt.ylim([-11, 0])
        # plt.show()

        end = time.time()
        print(f"Time to calculate LE with 2 samples: {end - start}")
        LEs_avg = torch.mean(LE_list, dim=0)
    return total_loss / (len(data_source) - 1), LEs_avg

def cal_LEs_from_trained_model(args, model, val_data, corpus, trial_num=None):
    criterion = nn.CrossEntropyLoss()
    # global
    # path_models_des = f"/home/ws8/caleb/dataset/PTB/models/stacked_LSTM_pruned"
    # path_LEs_des = f"/home/ws8/caleb/dataset/PTB/LEs/stacked_LSTM_pruned"

    # local
    # path_models_des = f"../models/RigL"
    # path_LEs_des = f"../LEs/RigL"
    path_models_des = f'../models/stacked_LSTM_pruned'
    path_LEs_des = f'../LEs/stacked_LSTM_pruned'

    # Load the best saved model.
    for i in range(100, 101):
        start = time.time()
        path_saved = f"{path_models_des}/___e{i}___{trial_num}.pt"
        if not os.path.exists(path_saved):
            continue
        else:
            with open(path_saved, 'rb') as f:
                saved = torch.load(path_saved)
                # model = saved['model']
                model.load_state_dict(saved['model_state_dict'])
                val_loss, LEs_avg = evaluate(args, model, val_data, corpus, criterion)
                print('=' * 89)
                print(f'| trial_num {trial_num} | At epoch {i} | val loss {val_loss:5.2f} | val ppl {math.exp(val_loss):8.2f}')
                print('=' * 89)

                LEs_stats = {}
                LEs_stats['LEs'] = LEs_avg
                LEs_stats['current_loss'] = val_loss
                LEs_stats['current_perplexity'] = math.exp(val_loss)

                if not os.path.exists(path_LEs_des):
                    os.mkdir(path_LEs_des)
                pickle.dump(LEs_stats, open(f'{path_LEs_des}/___e{i}___{trial_num}.pickle', 'wb'))
        end = time.time()
        print(f"Time elpased: {end - start} s")


def LE_main(args):
    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)

    val_data = batchify(corpus.valid, args.eval_batch_size).to(args.device)
    # val_data = batchify(corpus.test, args.eval_batch_size).to(args.device)

    model = Stacked_LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(args.device)
    cal_LEs_from_trained_model(args=args, model=model, val_data=val_data, corpus=corpus, trial_num=args.trial_num)

if __name__ == "__main__":
    starting_ind = 14000
    max_evals = 20
    remaining_indices = np.array(range(starting_ind, starting_ind + max_evals))
    a = len(remaining_indices)
    # trials = pickle.load(open(f'../trials/PPL_tpe_trials_num_20.pickle', 'rb'))
    trials = pickle.load(open(f'../trials/PPL_tpe_trials_num_{max_evals}_ind_{starting_ind}.pickle', 'rb'))

    trial = trials.trials[16]
    args = trial['result']['args']
    args.data = '/home/ws8/caleb/dataset/PTB/penn/'
    print(args)
    # args = Args().args
    args.trial_num = 14016
    args.eval_batch_size = 2
    args.evaluate = '../models/stacked_LSTM_pruned/___e0___14008.pt'
    LE_main(args)
    # starting_idx = 100
    # num_trials = 1
    # for trial_num in range(starting_idx, starting_idx + num_trials):
    #     LE_main(trial_num)
    # LE = pickle.load(open('../LEs/___e7___200.pickle', 'rb'))
    # print(LE)
