import torch
from AE_utils import mini_batch_ae, train_val_split
import os
import util
from util import *
from SMNIST_AEPredNet import AEPredNet


def ae_train(model, train_data, train_targets, val_data, val_targets,
             batch_size=64, val_batch_size=64, alpha=1,
             epochs=100, verbose=True, print_interval=10,
             suffix='', device=torch.device('cpu'), inputs_epoch=None):
    train_loss = torch.zeros((epochs,), device=device)
    val_loss = torch.zeros((epochs,), device=device)
    val1 = torch.zeros((epochs,), device=device)
    val2 = torch.zeros((epochs,), device=device)
    for epoch in range(epochs):
        x_train = mini_batch_ae(train_data, batch_size)
        tar_train = mini_batch_ae(train_targets, batch_size)
        # print("tar_train: ", tar_train)
        x_val = mini_batch_ae(val_data, val_batch_size)
        tar_val = mini_batch_ae(val_targets, val_batch_size)
        tl = 0
        vl = 0
        vl1 = 0
        vl2 = 0
        train_batches = 0
        val_batches = 0
        train_samples_total = 0
        val_samples_total = 0
        for xt, tt in zip(x_train, tar_train):
            train_samples = xt.shape[0]
            train_samples_total += train_samples
            loss, outputs = model.train_step_ae(inputs=xt.to(device), targets=tt.to(device), alpha=alpha)
            # if epoch % print_interval == 0:
            tl += float(loss * train_samples)
        for xv, tv in zip(x_val, tar_val):
            val_samples = xv.shape[0]
            val_samples_total += val_samples
            losses = model.val_step_ae(input=xv.to(device), targets=tv.to(device), alpha=alpha)
            vl += float(losses[0]) * val_samples
            vl1 += float(losses[1]) * val_samples
            vl2 += float(losses[2]) * val_samples

        train_loss[epoch] = tl / train_samples_total
        val_loss[epoch] = vl / val_samples_total
        # print(torch.mean(torch.Tensor(vl1)).shape)
        val1[epoch] = vl1 / val_samples_total
        val2[epoch] = vl2 / val_samples_total

        if val2[epoch] < model.best_val:
            model.best_val = val2[epoch]
            model.best_state = model.state_dict()
        model.global_step += 1
        if verbose and (epoch+1) % print_interval == 0:
            # print(model.val_loss)
            print(f'Validation Loss at epoch {model.global_step}: {val_loss[epoch]:.3f}')
            print(f'reconstruction Loss at epoch {model.global_step}: {val1[epoch]:.3f}')
            print(f'Best Prediction Loss: {model.best_val:.3f}')
    #         for i, (name, param) in enumerate(model.named_parameters()):
    #             if i == 0:
    #                 print(name, param[0][0])
    # print(val2)
    model.train_loss = torch.cat((model.train_loss, train_loss))
    model.val_loss = torch.cat((model.val_loss, val_loss))
    model.vl1 = torch.cat((model.vl1, val1))
    model.vl2 = torch.cat((model.vl2, val2))
    model.alphas = torch.cat((model.alphas, torch.Tensor([alpha, epochs]).unsqueeze(dim=0)), dim=0)

    return model
