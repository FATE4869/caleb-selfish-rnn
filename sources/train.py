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


def model_save(model, fn, epoch=None):
    if epoch is not None:
        fn = f"../models/___e{epoch}___" + fn
    torch.save(model.state_dict(), fn)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt_len):
    # bptt_len = args.bptt
    seq_len = min(bptt_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def train_main(args, device):
    torch.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################
    corpus = data.Corpus(args.data)
    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)
    ntokens = len(corpus.dictionary)

    ###############################################################################
    # Build the model
    ###############################################################################

    if args.model == 'RHN':
        model = RHN(vocab_sz=ntokens, embedding_dim=args.emsize, hidden_dim=args.nhid,
                                 recurrence_depth=args.nrecurrence_depth, num_layers=args.nlayers, input_dp=args.dropouti,
                                 output_dp=args.dropout, hidden_dp=args.dropouth, embed_dp=args.dropoute,
                                 tie_weights=args.tied, couple=args.couple).to(device)

    elif args.model == 'LSTM':
        model = Stacked_LSTM(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
        # model= nn.DataParallel(model,device_ids = [0, 1])
        model.to(device)
    criterion = nn.CrossEntropyLoss()

    # if args.fromEpoch is not None:
    #     model_path = f'../models/___e{args.fromEpoch}___{args.save}'
    #
    #         # '../../../dataset/PTB/models/stacked_LSTM_pruned/___e99___320.pt'
    #     model.load_state_dict(torch.load(model_path))

    ###############################################################################
    # Train and evaluate code
    ###############################################################################
    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, bptt_len=args.bptt)

                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def train(mask=None):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)

        if args.model != 'Transformer':
            hidden = model.init_hidden(args.batch_size)
        # train_data = [n, batch_size]
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            # data = [args.bptt, batch_size], targets = [args.bptt * batch_size]
            data, targets = get_batch(train_data, i, args.bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            optimizer.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if mask is None:
                optimizer.step()
            else:
                mask.step()
            total_loss += loss.item()
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                sys.stdout.flush()
    ###############################################################################
    # Evaluating
    ###############################################################################
    if args.evaluate:
        print("=> loading checkpoint '{}'".format(args.evaluate))
        model.load_state_dict(torch.load(args.evaluate))
        print('=> testing...')
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| Final test | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        sys.stdout.flush()
    ###############################################################################
    # Training
    ###############################################################################
    else:
        mask = None
        lr = args.lr
        best_val_loss = []
        stored_loss = 100000000
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            optimizer = None
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
    #
            mask = None
            if args.sparse:
                decay = CosineDecay(args.death_rate, args.epochs * len(train_data) // args.bptt)
                mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                               redistribution_mode=args.redistribution, model=args.model, args=args)
                mask.add_module(model, sparse_init=args.sparse_init, density=args.density)
            # print(mask)
            model_save(model, args.save, epoch=0)
            # Loop over epochs.
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train(mask)
                # what's this t0? For averaging?
                if 't0' in optimizer.param_groups[0]:
                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()
                    val_loss2 = evaluate(val_data)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                    print('-' * 89)

                    if val_loss2 < stored_loss:
                        model_save(model, args.save, epoch=epoch)
                        print('Saving Averaged!')
                        stored_loss = val_loss2

                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()

                    if args.sparse and epoch < args.epochs + 1:
                        mask.at_end_of_epoch(epoch)

                else:
                    val_loss = evaluate(val_data)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                          'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                    print('-' * 89)

                    if val_loss < stored_loss:
                        model_save(model, args.save, epoch=epoch)
                        print('Saving model (new best validation)')
                        stored_loss = val_loss

                    if args.optimizer == 'adam':
                        scheduler.step(val_loss)

                    if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                            len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                        print('Switching to ASGD')
                        optimizer = Sparse_ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                        mask.optimizer = optimizer
                        mask.init_optimizer_mask()

                    if args.sparse and 't0' not in optimizer.param_groups[0]:
                        mask.at_end_of_epoch(epoch)

                    best_val_loss.append(val_loss)

                print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')




        # # Load the best saved model.
        # with open(args.save, 'rb') as f:
        #     model.load_state_dict(torch.load(args.save))
        #
        # # Run on test data.
        # test_loss = evaluate(test_data)
        # print('=' * 89)
        # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        #     test_loss, math.exp(test_loss)))
        # print('=' * 89)
        # sys.stdout.flush()
if __name__ == '__main__':
    init = 'uniform'
    optimizer = 'sgd'

    for i, seed in enumerate(range(190, 190 + 1)):
        train_main(seed, init=init, opt=optimizer)
