import torch
from AEPredNet import AEPredNet
from AE_utils import mini_batch_ae, train_val_split
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

# util in training

def train_data_loading(N_s, a_s='random', inputs_epochs=None, interpolated=True, isTrain=True):
    inputs_Ns_as_es = torch.tensor([])
    targets_Ns_as_es = torch.tensor([])
    for i, inputs_epoch in enumerate(inputs_epochs):
        for N in N_s:
            if a_s == 'random':
                if isTrain:
                    folder_name = f'TrainingData/interpreted/N_{N}/a_random/train/'
                else:
                    folder_name = f'TrainingData/interpreted/N_{N}/a_random/test/'
                file_name = f'epoch_{inputs_epoch}.pickle'
                data = pickle.load(open(folder_name + file_name, 'rb'))
                inputs, targets = data['inputs'], data['targets']
                inputs_Ns_as_es = torch.cat((inputs_Ns_as_es, inputs), dim=0)
                targets_Ns_as_es = torch.cat((targets_Ns_as_es, targets), dim=0)
            else:
                for a in a_s:
                    folder_name = f'TrainingData/interpreted/N_{N}/a_{a}/'
                    file_name = f'epoch_{inputs_epoch}.pickle'
                    data = pickle.load(open(folder_name + file_name, 'rb'))
                    inputs, targets = data['inputs'], data['targets']
                    inputs_Ns_as_es = torch.cat((inputs_Ns_as_es, inputs), dim=0)
                    targets_Ns_as_es = torch.cat((targets_Ns_as_es, targets), dim=0)
    return inputs_Ns_as_es, targets_Ns_as_es

def data_loading(N, a, model_type):
    file_path = '../lyapunov-hyperopt-master/SMNIST/trials/{}_{}_uni_{}.pickle'.format(model_type, N, a)
    trials = pickle.load(open(file_path, 'rb'))
    return trials

def data_path_name(N, a_s, interpreted=True, inputs_epoch=None, target_epoch=None, result=False):
    if len(a_s) > 1:
        a_name = 'a_mixed'
    else:
        a_name = 'a_{}'.format(a_s[0])

    N_name = 'N_{}'.format(N)

    if interpreted:
        i_name = 'interpreted'
    else:
        i_name = 'non_interpreted'

    folder_name = 'TrainingData/' + i_name + '/' + N_name + '/' + a_name + '/'
    file_name = f'epoch_{inputs_epoch}.pickle'
    return folder_name, file_name


def result_path_name(N_s, a_s, inputs_epochs=None, target_epoch=None):
    if len(a_s) > 1:
        a_name = 'a_mixed'
    else:
        a_name = 'a_{}'.format(a_s[0])

    if len(N_s) > 1:
        N_name = 'N_mixed'
    else:
        N_name = 'N_{}'.format(N_s[0])

    if len(inputs_epochs) > 1:
        ie_name = 'epochs'
    else:
        ie_name = f"epoch_{inputs_epochs[0]}"

    folder_name = 'Results/' + N_name + '/' + a_name + '/'
    file_name = ie_name + '.pickle'
    return folder_name, file_name


def data_saving(data, N_s=None, a_s=None, interpreted=None, inputs_epoch=None, target_epoch=None, folder_name=None, file_name=None):
    # print(inputs_as.shape, targets_as.shape)
    # Save data for AE network
    if (folder_name is None) or (file_name is None):
        folder_name, file_name = data_path_name(N_s, a_s, interpreted, inputs_epoch, target_epoch=None)
    path_name = folder_name + file_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    pickle.dump(data, open(path_name, 'wb'))


def model_saving(model, N_s=None, a_s=None, inputs_epochs=None):
    result_folder_name, _ = result_path_name(N_s, a_s, inputs_epochs)
    result_folder_name += (model.prediction_loss_type + '/')

    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)
    torch.save(model, result_folder_name + f'/ae_prednet_{model.global_step}.ckpt')


# Visualizing g vs. target_learning loss(and its average)
def LEs_plotting(inputs, a=None, input_epoch=None, trials2display=5):
    plt.figure()
    num_trials, N = inputs.shape

    for i in range(trials2display):
        x_axis = range(N)
        plt.scatter(x_axis, inputs[i, :])

    plt.title(f'Display the first {trials2display} trials with a: {a}, N: {N},'
              f'input epoch: {input_epoch}')
    # x_axis = gs[0] * torch.ones_like(targets_all)
    # plt.scatter(x_axis, targets_all,s = 1)
    # plt.scatter(gs, targets_avg)
    # plt.plot(gs, targets_avg, 'k-', linewidth=3)
    # plt.axis([min(gs)-0.1, max(gs) + 0.1, -.1, 1.1])
    plt.show()


def checkNan(inputs, targets, replace=False):
    [m, n] = inputs.shape
    if replace:
        new_inputs, new_targets = torch.ones_like(inputs), torch.ones_like(targets)
        count = 0
        for i in range(m):
            if (sum(torch.isnan(inputs[i, :])) > 0):
                print(i)
            else:
                new_inputs[count, :] = inputs[i, :]
                new_targets[count] = targets[i]
                count += 1
        new_inputs = new_inputs[:count, :]
        new_targets = new_targets[:count]
        return new_inputs, new_targets
    else:
        for i in range(m):
            if (sum(torch.isnan(inputs[i, :])) > 0):
                print(i)


# interpolate the inputs so that its dimension increases to twice
def interpolate(inputs, inputs_dim=512, target_dim=1024):
    m, n = inputs.shape
    device = inputs.device
    shift = torch.cat((inputs[:, 1:], torch.zeros((m, 1), dtype=inputs.dtype)), dim=1).to(device)

    diffs = (inputs - shift) / 2;
    diffs[:, -1] = diffs[:, -2]

    new_inputs = torch.zeros(m, 0).to(device)
    for col, diff in zip(inputs.T, diffs.T):
        new_inputs = torch.cat((new_inputs, col.unsqueeze(1)), dim=1)
        new_inputs = torch.cat((new_inputs, (col - diff).unsqueeze(1)), dim=1)

    # check if it works
    # print(inputs[0,:4])
    # print(new_inputs[0,:4])
    return new_inputs

def normalization(inputs):
    # print(torch.max(inputs), torch.min(inputs), inputs.shape)
    inputs_max = torch.max(inputs)
    inputs_min = torch.min(inputs)
    inputs = (inputs - inputs_min) / (inputs_max - inputs_min)
    # inputs -= torch.max(inputs)
    # print(torch.max(inputs), torch.min(inputs))
    return  inputs