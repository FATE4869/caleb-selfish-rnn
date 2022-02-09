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
from sources.models import Stacked_LSTM
from sources.lyapunov import calc_LEs_an
import os
from sources import data


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

        LEs_avg = torch.mean(LE_list, dim=0)
    return total_loss / (len(data_source) - 1), LEs_avg

def cal_LEs_from_trained_model(args, model, val_data, corpus, trial_num=None, target_epoch=None):
    criterion = nn.CrossEntropyLoss()
    # # Load the best saved model.
    for i in range(0, 61):
        start = time.time()
        path_saved = f"models/stacked_LSTM_pruned/___e{i}___{trial_num}.pt"
        if not os.path.exists(path_saved):
            continue
        else:
            with open(path_saved, 'rb') as f:
                model.load_state_dict(torch.load(path_saved))
                val_loss, LEs_avg = evaluate(args, model, val_data, corpus, criterion)
                print('=' * 89)
                print(f'| trial_num {trial_num} | At epoch {i} | val loss {val_loss:5.2f} | val ppl {math.exp(val_loss):8.2f}')
                print('=' * 89)

                LEs_stats = {}
                LEs_stats['LEs'] = LEs_avg
                LEs_stats['current_loss'] = val_loss
                LEs_stats['current_perplexity'] = math.exp(val_loss)
                if not os.path.exists('LEs/stacked_LSTM'):
                    os.mkdir('LEs/stacked_LSTM/')
                pickle.dump(LEs_stats, open(f'LEs/stacked_LSTM_pruned/___e{i}___{trial_num}.pickle', 'wb'))
        end = time.time()
        print(f"Time elpased: {end - start} s")
def checkLEs(inputs_epoch, N, model_type, trials_num):
    LEs = pickle.load(open(f'trials/lstm/LEs/e_{inputs_epoch}/{model_type}_{N}_trials_{trials_num}.pickle','rb'))
    print(f'N: {N}, trials_num: {trials_num}, inputs_epoch: {inputs_epoch}, length: {len(LEs)}, keys are: {LEs[0].keys()}')
    print()

def main():
    # for gru models
    # N_s = [64]
    # a_s = [2.0, 2.2, 2.5, 2.7, 3.0]
    # for N in N_s:
    #     for a in a_s:
    #         cal_LEs_from_trained_model(N, a)

    # for lstm models
    parser = argparse.ArgumentParser(description='PyTorch implementation of Selfish-RNN')
    args = parser.parse_args()
    args.data = 'data/penn'
    args.model = 'LSTM'
    args.emsize = 1500
    args.nhid = 1500
    args.nlayers = 2
    args.dropout = 0.65
    args.tied = False
    args.cuda = True
    args.eval_batch_size = 10
    args.bptt = 35
    # torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    # val_data = batchify(corpus.valid, args.eval_batch_size).to(device)
    val_data = batchify(corpus.test, args.eval_batch_size).to(device)
    model = Stacked_LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    target_epoch = 30
    for trial_num in range(173, 173 + 4):
        cal_LEs_from_trained_model(args=args, model=model, val_data=val_data, corpus=corpus, trial_num=trial_num, target_epoch=target_epoch)

if __name__ == "__main__":
    main()
