import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def perplexity_compare(names, indices, num_epochs, labels=None):
    if labels is None:
        labels = names
    plt.figure()
    for i, name in enumerate(names):
        print(i, names)
        ppls = []
        previous_ppl = 0
        for epoch in range(num_epochs):
            file_path = f'../../dataset/PTB/{name}/___e{epoch}___{indices[i]}.pickle'
            if os.path.exists(file_path):
                LEs = pickle.load(open(file_path, 'rb'))
                # print(LEs)
                ppls.append(LEs['current_perplexity'])
                previous_ppl = LEs['current_perplexity']
            else:
                ppls.append(previous_ppl)
        print(ppls)
        plt.plot(range(6, len(ppls)), ppls[6: ], label=labels[i])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("perplexity")
    plt.show()

def LE_loading(num_epochs, folder_name, trial_index):
    for i, epoch in enumerate(range(num_epochs + 1)):
        file_name = f'../../dataset/PTB/{folder_name}/___e{epoch}___{trial_index}.pickle'

        if os.path.exists(file_name):
            # print(file_namename)
            LE = pickle.load(open(file_name, 'rb'))
            LE = torch.transpose(torch.unsqueeze(LE['LEs'], 1), 0, 1)
            if i == 0:
                LEs = LE
            else:
                LEs = torch.concat([LEs, LE])
    # print(LEs)
    LEs = LEs.detach().cpu()
    return LEs

def tsne(X, dim=2, divider_indices=None, names=None, labels=None, use_tsne=True):
    if labels is None:
        labels = names
    z = torch.zeros((), device=X.device, dtype=X.dtype)
    inf_indices = torch.where(X == -torch.inf)
    for i, inf_index in enumerate(range(inf_indices[0].shape[0])):
        X[inf_indices[0][i], inf_indices[1][i]] = 0
    # PCA
    if use_tsne:
        tsne_model = TSNE(perplexity=10, n_components=2, random_state=1)
        x_embedded = tsne_model.fit_transform(X.detach().numpy())
    else: # using PCA
        U,S,V = torch.pca_lowrank(X)
        x_embedded = torch.matmul(X, V[:, :dim]).detach().numpy()
    # T-SNE


    fig = plt.figure()
    if (dim==2):
        plt.figure(1)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=20, c='black')
        if divider_indices is None:
            plt.plot(x_embedded[:, 0], x_embedded[:, 1])
            # plt.scatter(x_embedded[6, 0], x_embedded[6, 1], s=100, c='orange')
            # plt.scatter(x_embedded[5, 0], x_embedded[5, 1], s=100 , c='orange')
            plt.scatter(x_embedded[0, 0], x_embedded[0, 1], s=100, c='blue')
        else:
            for i in range(len(divider_indices) - 1):
                plt.scatter(x_embedded[divider_indices[i], 0],
                            x_embedded[divider_indices[i], 1], s=100, c='blue')
                plt.scatter(x_embedded[divider_indices[i + 1] - 1, 0],
                            x_embedded[divider_indices[i + 1] - 1, 1], s=100, c='green')
                plt.plot(x_embedded[divider_indices[i]: divider_indices[i+1], 0],
                         x_embedded[divider_indices[i]: divider_indices[i+1], 1],
                         label=labels[i])
            plt.legend()
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1],
                   x_embedded[:, 2], s=6)
    plt.legend()
    if use_tsne:
        plt.title("t-sne")
    else:
        plt.title("PCAs")
    plt.show()
    print(x_embedded.shape)

    # tsne_results = {'x_embedded':x_embedded, 'targets':targets}
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne.png', dpi=200)

    # return tsne_results
def main():
    # perplexity_compare()
    names = [
            'LEs/stacked_LSTM_full',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            ]
    indices = [
               141,
               163,
               165,
               167,
               169,
               173,
               171,
               175
               ]
    labels = [
              'pr = 0',
              'pr = 0.2', #'pr = 0.2',
              'pr = 0.4', #'pr = 0.4',
              'pr = 0.6', #'pr = 0.6',
              'pr = 0.8', #'pr = 0.8',
              'pr = 0.85',
              'pr = 0.9', #'pr = 0.9'
              'pr = 0.95'
              #
              ]
    # labels = None
    num_epochs = 30
    perplexity_compare(names, indices, num_epochs, labels=labels)
    sum = 0
    divider_indices = [0]
    for i, name in enumerate(names):
        print(i, name, indices[i])
        LE = LE_loading(num_epochs=60, folder_name=name, trial_index=indices[i])
        sum += LE.shape[0]
        # plt.figure(2)
        # colors = ['blue', 'green', 'purple']

        # for j in range(3):
        # for j in range(LE.shape[0]):
        #     if j == 0:
        #         plt.scatter(range(len(LE[0])), LE[j], s=10, label='before training')#, c=colors[j],)
        #     else:
        #         plt.scatter(range(len(LE[0])), LE[j], s=10, label=f'epoch{j}')# c=colors[j])
        #     plt.xlabel('index')
        # # plt.legend()
        # plt.ylabel('LE')
        # plt.ylim([-50, 0])
        # plt.title(f'{labels[i]}')
        # plt.show()
        if i == 0:
            LEs = LE
        else:
            LEs = torch.concat([LEs, LE])
        divider_indices.append(sum)
    print(divider_indices)
    tsne(LEs, dim=2, divider_indices=divider_indices, names=names, labels=labels, use_tsne=True)
if __name__ == '__main__':
    main()