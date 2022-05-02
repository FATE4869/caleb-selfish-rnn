import os
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def LE_loading(folder_name, trial_index, num_epochs=None, epochs=None):
    if num_epochs is not None:
        epochs = range(num_epochs)
    for i, epoch in enumerate(epochs):
        file_name = f'/home/ws8/caleb/dataset/PTB/{folder_name}/___e{epoch}___{trial_index}.pickle'

        if os.path.exists(file_name):
            # print(file_namename)
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

def nextRound(remaining_indices, ref_indices, current_epoch, use_trained_ref=False):
    # load reference LE
    if use_trained_ref:
        LEs = LE_loading(folder_name='LEs/stacked_LSTM_full', trial_index=ref_indices,
                         num_epochs=100)
    else:
        LEs = LE_loading(folder_name='LEs/stacked_LSTM_full', trial_index=ref_indices,
                         num_epochs=current_epoch)
    ref_len = LEs.shape[0]
    for i, ind in enumerate(remaining_indices):
        LE = LE_loading(folder_name='LEs/stacked_LSTM_pruned', trial_index=ind,
                        num_epochs=current_epoch)
        LEs = torch.concat([LEs, LE])
    # LE = torch.rand([current_epoch * len(remaining_indices) + 4, 10])

    divider_indices = np.array([0])
    divider_indices = np.concatenate([divider_indices,
                                      np.arange(ref_len, ref_len + len(remaining_indices) * current_epoch, current_epoch)])
    print(divider_indices)
    x_embedded = tsne(LEs, use_tsne=False)
    distances = np.zeros(len(remaining_indices))
    for ind in range(len(remaining_indices)):
        # for epoch in range(current_epoch):
        distances[ind] = np.sqrt(
            np.sum(np.square(x_embedded[divider_indices[1] - 1, :]
                             - x_embedded[current_epoch + divider_indices[ind+1] - 1, :])))

    print(distances)
    indices_sorted = np.argsort(distances)
    num_remaining_trials = int(len(remaining_indices) / 2)
    remaining_indices = np.array(remaining_indices)
    return remaining_indices[indices_sorted[:num_remaining_trials]].tolist()

def competition(starting_epoch, incremental_epoch, trial_indices, ref_indices, use_trained_ref=False):
    current_epoch = starting_epoch
    remaining_indices = trial_indices
    count = 0
    while len(remaining_indices) > 3:
        print(f"-------------Round {count} --------")
        print(f"current epoch: {current_epoch}")
        print(f'current trial indices: {remaining_indices}')
        remaining_indices = nextRound(remaining_indices, ref_indices, current_epoch, use_trained_ref)
        print(f'remaining trial indices: {remaining_indices}')
        print(f"-------------Round {count} --------")
        print("\n")
        count += 1
        current_epoch += incremental_epoch

def tsne(X, dim=2, trial_indices=None, divider_indices=None, names=None, labels=None, use_tsne=True, limited_samples=None):
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
    return x_embedded


def main():
    trial_indices = np.linspace(5024, 5071, 48, dtype=int).tolist()
    competition(starting_epoch=3, incremental_epoch=2, trial_indices=trial_indices, ref_indices=161, use_trained_ref=True)
if __name__ == '__main__':
    main()