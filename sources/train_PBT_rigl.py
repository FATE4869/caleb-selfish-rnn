from __future__ import print_function
import math
import time
import torch
import torch.nn as nn
from rigl_torch.RigL import RigLScheduler
from sources.models import RHN, Stacked_LSTM
from args import Args
from dataloader import PBT_Dataloader
from torch.optim.lr_scheduler import StepLR
from sources.Sparse_ASGD import Sparse_ASGD
from sources.LE_calculation import LE_main

def model_save(model, fn, epoch=None):
    if epoch is not None:
        fn = f"../models/RigL/___e{epoch}___" + fn
    torch.save(model.state_dict(), fn)

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
        # if pruner():
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'| Train Epoch: {epoch} | [{batch_idx}/{num_iterations} | '
                  f'({batch_idx / num_iterations * 100.0:.2f}%)] | '
                  f'\tLoss: {loss.item():.6f}| ppl {math.exp(loss.item()):8.2f} | '
                  f' bpc {loss.item() / math.log(2):8.3f} |')


def evaluate(args, model, device, pbt_dataloader, criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(args.eval_batch_size)
    ntokens = len(pbt_dataloader.corpus.dictionary)
    total_loss = 0.
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, pbt_dataloader.val_data.size(0) - 1, args.bptt)):
            samples, targets = pbt_dataloader.get_batch(pbt_dataloader.val_data, i)
            samples, targets = samples.to(device), targets.to(device)
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(samples, hidden)
            output_flat = outputs.view(-1, ntokens)
            total_loss += len(samples) * criterion(output_flat, targets).item()
    return total_loss / (pbt_dataloader.val_data.size(0))

def main(args):

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
    optimizer = Sparse_ASGD(model.parameters())
    # optimizer.step()
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    pruner = lambda: True
    if args.sparsity > 0:
        T_end = int(0.75 * args.epochs * pbt_dataloader.train_loader_len)
        pruner = RigLScheduler(model, optimizer, sparsity=args.sparsity, alpha=args.alpha,
                               delta=args.delta, static_topo=args.static_topo, T_end=T_end,
                               ignore_linear_layers=False, grad_accumulation_n=args.grad_accumulation_n)

    # model_save(model, args.save, epoch=0)
    stored_loss = 100000000
    print(args)
    for epoch in range(0, args.epochs):

        epoch_start_time = time.time()
        print(pruner)
        train(args, model, device, pbt_dataloader, optimizer, criterion, pruner=pruner, epoch=epoch)

    #     val_loss = evaluate(args, model, device, pbt_dataloader, criterion)
    #     print('-' * 89)
    #     print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | valid loss {val_loss:5.2f} | '
    #           f'valid ppl {math.exp(val_loss):8.2f} | valid bpc {val_loss / math.log(2):8.3f}')
    #     print('-' * 89)
    #     scheduler.step()
    #     if val_loss < stored_loss:
    #         model_save(model, args.save, epoch=epoch)
    #         print('Saving model (new best validation)')
    #         stored_loss = val_loss

if __name__ == '__main__':
    args = Args().args
    args.sparsity = 0
    args.save = '1000.pt'
    args.epochs = 50

    args.delta = 100
    args.alpha = 0.3
    args.static_topo = False  # if 1, use random sparisty topo and remain static, else 0
    args.grad_accumulation_n = 1
    deltas = [100]
    alphas = [0.4]
    count = 0
    args.eval_batch_size = 2
    for delta in deltas:
        for alpha in alphas:
            args.delta = delta
            args.alpha = alpha
            args.save = f'{1010 + count}.pt'
            main(args)
            # args.trial_num = 1009 + count
            # LE_main(args)
            count += 1