import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from numpy.linalg import norm

def return_candidates(distances, num_trials, starting_epoch=3, incremental_epoch=2):
    distance2ind = {}
    distances_check = distances
    for i in range(num_trials):
        distance2ind[distances_check[i, -1]] = i
    checking_epoch = starting_epoch
    # indices_original = np.linspace(0, 23, 24, dtype=int)
    indices_remaining = np.linspace(0, num_trials-1, num_trials, dtype=int)
    while len(indices_remaining) > 3:
        # indices_sorted = np.argsort(distances_check[:, checking_epoch])
        indices_sorted = np.argsort(distances_check[:, checking_epoch])
        remaining_trials = int(len(indices_sorted) / 2)
        indices_remaining = indices_sorted[:remaining_trials]
        distances_check = distances_check[indices_remaining]
        checking_epoch += incremental_epoch
    indices_candidates = []
    for distance in distances_check[:, -1]:
        indices_candidates.append(distance2ind[distance])
    return indices_candidates

def cal_distance(data, divider_indices, num_trials, num_epochs, starting_epoch=0):
    distances = np.zeros([num_epochs, num_trials])
    plt.figure(1)
    for a in range(num_epochs):
        for j in range(num_trials):
            distances[a, j] = torch.sqrt(torch.sum(torch.square(data[a + starting_epoch]
                                                       - data[a + divider_indices[j+1] + starting_epoch])))

        plt.plot(range(9), distances[a, :], label=f'{a + starting_epoch}')
    plt.legend()
    plt.show()


def perplexity_compare(names, indices, num_epochs, labels=None, indices_candidates=[]):
    if labels is None:
        labels = names
    # colors = cm.rainbow(np.linspace(0, 1, len(names)))
    # print(colors)
    colors = cm.rainbow(np.linspace(0, 1, 4))
    ppls = torch.zeros([len(names), num_epochs])
    divider_indices = []
    plt.figure()
    count = 0
    for i, name in enumerate(names):
        # print(i, name)
        previous_ppl = 0
        for epoch in range(num_epochs):
            file_path = f'/home/ws8/caleb/dataset/PTB/{name}/___e{epoch}___{indices[i]}.pickle'
            if os.path.exists(file_path):
                LEs = pickle.load(open(file_path, 'rb'))
                # print(LEs)
                ppls[i, epoch] = LEs['current_perplexity']
                previous_ppl = LEs['current_perplexity']
            else:
                ppls[i, epoch] = previous_ppl

        divider_indices.append(num_epochs * i)
        # if i in indices_candidates:
        #     plt.plot(range(6, num_epochs), ppls[i, 6: num_epochs], label=labels[i], color=colors[count])
        #     count += 1
        # else:
        #     plt.plot(range(6, num_epochs), ppls[i, 6: num_epochs], label=labels[i], color='k')

    # plt.legend()
    # plt.xlabel("epoch")
    # plt.ylabel("perplexity")
    print(ppls[:, -1])
    # print(ppls[indices_candidates, -1])
    # plt.show()
    return_candidates(ppls[1:].numpy(), num_trials=24, starting_epoch=4, incremental_epoch=2)
    # cal_distance(torch.flatten(ppls), divider_indices, num_trials=len(names)-1, num_epochs=10, starting_epoch=0)


def LE_loading(folder_name, trial_index, num_epochs=None, epochs=None):
    if num_epochs is not None:
        epochs = range(num_epochs)
    for i, epoch in enumerate(epochs):
        file_name = f'/home/ws8/caleb/projects/caleb-selfish-rnn/{folder_name}/___e{epoch}___{trial_index}.pickle'
        # file_name = f'/home/ws8/caleb/dataset/PTB/{folder_name}/___e{epoch}___{trial_index}.pickle'

        if os.path.exists(file_name):
            LE = pickle.load(open(file_name, 'rb'))
            LE = torch.transpose(torch.unsqueeze(LE['LEs'], 1), 0, 1)
            previous_LE = LE
            if i == 0:
                LEs = LE
            else:
                LEs = torch.concat([LEs, LE])
        else:
            LEs = torch.concat([LEs, previous_LE])
    # print(LEs)
    LEs = LEs.detach().cpu()
    return LEs

def tsne(X, dim=2, divider_indices=None, names=None, labels=None, use_tsne=True, limited_samples=None):
    if labels is None:
        labels = names
    z = torch.zeros((), device=X.device, dtype=X.dtype)
    inf_indices = torch.where(X == -torch.inf)
    for i, inf_index in enumerate(range(inf_indices[0].shape[0])):
        X[inf_indices[0][i], inf_indices[1][i]] = 0
    # dimension reduction
    if use_tsne: # T-SNE
        tsne_model = TSNE(perplexity=10, n_components=2, random_state=1)
        x_embedded = tsne_model.fit_transform(X.detach().numpy())
    else: # PCA
        U,S,V = torch.pca_lowrank(X)
        x_embedded = torch.matmul(X, V[:, :dim]).detach().numpy()
    # l2 distance
    distance = np.sqrt(np.sum(
        np.square(x_embedded[divider_indices[1] - 1] - x_embedded[divider_indices[2] - 1])
    ))
    # cosine distance
    # distance = np.dot(x_embedded[divider_indices[1] - 1], x_embedded[divider_indices[2] - 1]) / \
    #            (norm(x_embedded[divider_indices[1] - 1]) * norm(x_embedded[divider_indices[2] - 1]))

    return distance
    # distances = np.zeros([len(divider_indices) - 1, divider_indices[1]])
    # distances[ind - 1, epoch] = np.sqrt(
    #     np.sum(np.square(x_embedded[epoch, :] - x_embedded[epoch + divider_indices[ind], :])))


def main():
    names = ['LEs/stacked_LSTM_full'] + ['LEs/stacked_LSTM_pruned'] * 24 # + ['LEs/RigL'] * 10
    # indices = [161, 5000, 5007, 5009, 5012, 5021, 5022]
    indices = [161] + np.linspace(5000, 5023, 24, dtype=int).tolist()
              # + np.linspace(1000, 1009, 10, dtype=int).tolist()

def LE_distance_main(trial_num, num_epochs, epoch=None, last_epoch_ref=False):

    folder_names = ['LEs/stacked_LSTM_full', 'LEs/stacked_LSTM_pruned']
    indices = [161, trial_num]
    divider_indices = [0]
    sum = 0
    for i, (name, index) in enumerate(zip(folder_names, indices)):
        if last_epoch_ref and i == 0:
            LE = LE_loading(name, index, num_epochs=None, epochs=[49])
        else:
            LE = LE_loading(name, index, num_epochs)
            # LE = LE_loading(name, index, num_epochs=None, epochs=[epoch])
        if i == 0:
            LEs = LE
        else:
            LEs = torch.cat([LEs, LE])
        sum += LE.shape[0]
        divider_indices.append(sum)

    distance = tsne(LEs, divider_indices=divider_indices, use_tsne=False)
    return distance



if __name__ == '__main__':
    # trials = pickle.load(open(f'../trials/tpe_trials_num6_inx6.pickle', 'rb'))
    for trial_num in range(6000, 6010):
        distance = LE_distance_main(trial_num)
        print(f'Trial Num: {trial_num}  --- > distance: {distance}')