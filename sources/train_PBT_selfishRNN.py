import argparse
import time
import math
import torch
import sys
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sources import data
from sources.sparse_rnn_core import Masking, CosineDecay
from sources.models import RHN, Stacked_LSTM
from sources.Sparse_ASGD import Sparse_ASGD
from selfish_rnn.selfish_core import SelfishScheduler
from LE_calculation import *
from dataloader import PBT_Dataloader

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


def train(args, model, device, pbt_dataloader, optimizer, criterion, pruner, epoch):
    model.train()
    hidden = model.init_hidden(args.batch_size)
    num_iterations = pbt_dataloader.train_loader_len
    for batch_idx, i in enumerate(range(0, pbt_dataloader.train_data.size(0) - 1, args.bptt)):
        samples, targets = pbt_dataloader.get_batch(pbt_dataloader.train_data, i)
        samples, targets = samples.to(device), targets.to(device)
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(samples, hidden)
        loss = criterion(outputs.view(-1, pbt_dataloader.ntokens), targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if pruner():
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'| Train Epoch: {epoch} | [{batch_idx}/{num_iterations} | '
                  f'({batch_idx / num_iterations * 100.0:.2f}%)] | '
                  f'\tLoss: {loss.item():.6f}| ppl {math.exp(loss.item()):8.2f} | '
                  f' bpc {loss.item() / math.log(2):8.3f} |')


def main():
    args = Args().args
    # args.sparsity = 0.6
    if args.sparsity <= 0.0:
        print('-------------------------------------------------------------------')
        print('heads up, RigL will not be used unless `--sparsity` is greater than 0!')
        print('-------------------------------------------------------------------')

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(args, device)

    # pbt_dataloader has attributes: train_data, val_data and test_data, each batchified according batch_size and
    # eval_batch_size, respectively
    pbt_dataloader = PBT_Dataloader(data_path=args.data, batch_size=args.batch_size,
                                    test_batch_size=args.eval_batch_size, bptt=args.bptt)

    model = Stacked_LSTM(args.model, pbt_dataloader.ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    criterion = nn.CrossEntropyLoss()

    pruner = lambda: True
    if args.sparsity > 0:
        T_end = int(0.75 * args.epochs * pbt_dataloader.train_loader_len)
        pruner = SelfishScheduler(model, optimizer, sparsity=args.sparsity, death_rate=args.death_rate, death_mode=args.death_mode,
                                  growth_mode=args.growth_mode, redistribution_mode=args.redistribution_mode)


        # pruner = RigLScheduler(model, optimizer, sparsity=args.sparsity, alpha=args.alpha,
        #                        delta=args.delta, static_topo=args.static_topo, T_end=T_end,
        #                        ignore_linear_layers=False, grad_accumulation_n=args.grad_accumulation_n)

    # model_save(model, args.save, epoch=0)
    stored_loss = 100000000
    for epoch in range(0, 1):
        epoch_start_time = time.time()
        # print(pruner)
        train(args, model, device, pbt_dataloader, optimizer, criterion, pruner=pruner, epoch=epoch)
        # val_loss = evaluate(args, model, device, pbt_dataloader, criterion)
        # print('-' * 89)
        # print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | valid loss {val_loss:5.2f} | '
        #       f'valid ppl {math.exp(val_loss):8.2f} | valid bpc {val_loss / math.log(2):8.3f}')
        # print('-' * 89)
        # if val_loss < stored_loss:
        #     model_save(model, args.save, epoch=epoch)
        #     print('Saving model (new best validation)')
        #     stored_loss = val_loss
if __name__ == '__main__':
    main()
