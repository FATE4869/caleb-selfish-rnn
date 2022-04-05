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


def perplexity_compare(names, indices, num_epochs, labels=None):
    if labels is None:
        labels = names
    colors = cm.rainbow(np.linspace(0, 1, len(names)))
    ppls = torch.zeros([len(names), num_epochs])
    divider_indices = []
    plt.figure()
    for i, name in enumerate(names):
        print(i, name)
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
        plt.plot(range(6, num_epochs), ppls[i, 6: num_epochs], label=labels[i], color=colors[i])
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("perplexity")
    print(ppls)
    plt.show()
    # cal_distance(torch.flatten(ppls), divider_indices, num_trials=len(names)-1, num_epochs=10, starting_epoch=0)


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

def tsne(X, dim=2, divider_indices=None, names=None, labels=None, use_tsne=True, limited_samples=None):
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

    colors = cm.rainbow(np.linspace(0, 1, len(divider_indices) - 1))

    distances = np.zeros([len(divider_indices) - 2, divider_indices[1]])
    # starting_epoch = 4
    plt.figure(1)
    for ind in range(1, len(divider_indices) - 1):
        for epoch in range(divider_indices[1]):
            distances[ind - 1, epoch] = np.sqrt(
                np.sum(np.square(x_embedded[epoch, :] - x_embedded[epoch + divider_indices[ind], :])))
        plt.scatter(range(divider_indices[1]), distances[ind - 1, :], label=labels[ind], color=colors[ind])
    plt.xlabel("epoch")
    plt.ylabel("distance")
    # plt.xlim([2.5, 4.5])
    # plt.xticks([3, 5])
    # plt.legend()
    plt.show()


    fig = plt.figure()
    if (limited_samples is None):
        plt.figure(1)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=20, c='black')
        if divider_indices is None:
            plt.plot(x_embedded[:, 0], x_embedded[:, 1])
            # plt.scatter(x_embedded[6, 0], x_embedded[6, 1], s=100, c='orange')
            # plt.scatter(x_embedded[5, 0], x_embedded[5, 1], s=100 , c='orange')
            plt.scatter(x_embedded[0, 0], x_embedded[0, 1], s=100, c='blue')
        else:
            for i in range(len(divider_indices) - 1):
                plt.scatter(x_embedded[divider_indices[i], 0], x_embedded[divider_indices[i], 1],
                            s=100, c='blue')
                plt.scatter(x_embedded[divider_indices[i+1]-1, 0], x_embedded[divider_indices[i+1]-1, 1],
                            s=100, c='green')
                # if i > 4:
                #     plt.plot(
                #              x_embedded[divider_indices[i]: divider_indices[i+1], 0],
                #              x_embedded[divider_indices[i]: divider_indices[i+1], 1],
                #             'r-', markersize=20,
                #              label=labels[i])
                # else:
                plt.plot(
                         x_embedded[divider_indices[i]: divider_indices[i+1], 0],
                         x_embedded[divider_indices[i]: divider_indices[i+1], 1],
                         color=colors[i],
                         label=labels[i])
            plt.legend()

    else:
        if divider_indices is None:
            plt.plot(x_embedded[:, 0], x_embedded[:, 1])
            # plt.scatter(x_embedded[6, 0], x_embedded[6, 1], s=100, c='orange')
            # plt.scatter(x_embedded[5, 0], x_embedded[5, 1], s=100 , c='orange')
            plt.scatter(x_embedded[0, 0], x_embedded[0, 0]+limited_samples, s=100, c='blue')
        else:
            for i in range(len(divider_indices) - 1):
                # scatter every point as black dot
                plt.scatter(x_embedded[divider_indices[i]: divider_indices[i] + limited_samples, 0],
                            x_embedded[divider_indices[i]: divider_indices[i] + limited_samples, 1],
                            s=20, c='black')
                # scatter starting point of each model as blue dot
                plt.scatter(x_embedded[divider_indices[i], 0], x_embedded[divider_indices[i], 1],
                            s=100, c='blue')

                # scatter ending point of each model as green dot
                plt.scatter(x_embedded[divider_indices[i] + limited_samples, 0], x_embedded[divider_indices[i] + limited_samples, 1],
                            s=100, c='green')

                # # scatter ending point of each model as green dot
                # plt.scatter(x_embedded[divider_indices[i+1]-1, 0], x_embedded[divider_indices[i+1]-1, 1],
                #             s=100, c='green')


                # if i > 4:
                #     plt.plot(
                #              # x_embedded[divider_indices[i]: divider_indices[i+1], 0],
                #              # x_embedded[divider_indices[i]: divider_indices[i+1], 1],
                #             x_embedded[divider_indices[i]: divider_indices[i] + limited_samples + 1, 0],
                #             x_embedded[divider_indices[i]: divider_indices[i] + limited_samples + 1, 1],
                #             'r-', markersize=20,
                #              label=labels[i])
                # else:
                plt.plot(
                        x_embedded[divider_indices[i]: divider_indices[i] + limited_samples + 1, 0],
                        x_embedded[divider_indices[i]: divider_indices[i] + limited_samples + 1, 1],
                        color=colors[i],
                        label=labels[i])
            plt.legend()



    if use_tsne:
        plt.title("t-sne")
    else:
        plt.title("PCAs")
    plt.show()
    print(x_embedded.shape)


def main():
    # perplexity_compare()
    names = [
            'LEs/stacked_LSTM_full',
            # 'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            #
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            # 'LEs/stacked_LSTM_pruned',
            ]
    indices = [
               161,
               # 141,
               # 163,
               # 165,
               # 167,
               # 169,
               # 173,
               # 171,
               # 175,
               # 177,
               #  300, 301, 302, 303, 304,
               #  305, 306, 307, 308, 309,
               #  310, 311, 312, 313, 314,
               #  315, 316, 317, 318, 319,
                340, 341, 342, 343,
                344, 345, 346, 347,
                # 328 ,329,
                # 330, 331, 332, 333, 334,
                # 335, 336, 337, 338, 339,
               # 201,
               # 202,
               # 203,
               # 204,
               # 205,
               # 206,
               # 207,
               # 208,
               ]
    labels = [
              'pr = 0',
              # 'pr = 0.2', #'pr = 0.2',
              # 'pr = 0.6 fix', #'pr = 0.4',
            'pr = 0.8 0',
            'pr = 0.8 1',
            'pr = 0.8 2',
            'pr = 0.8 3',
            'pr = 0.8 4',
            'pr = 0.8 5',
            'pr = 0.8 6',
            'pr = 0.8 7',
            # 'pr = 0.8 8',

            # 'pr = 0.8',' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
            # 'pr = 0.8', ' pr = 0.8', 'pr = 0.8', 'pr = 0.8', 'pr = 0.8',
              # 'pr = 0.80',
              # 'pr = 0.80',
              # 'pr = 0.9', #'pr = 0.9'
              # 'pr = 0.95',
              # 'pr = 0.6, unif, SNT-ASGD',

            # 'pr = 0.4, d = Mag, r = Mag',
            # 'pr = 0.6, d = Mag, r = Non-zeros',
            # 'pr = 0.6, d = Mag, r = none',
            # 'pr = 0.4, d = SET, r = Mag',
            # 'pr = 0.6, d = SET, r = Non-zeros',
            # 'pr = 0.6, d = SET, r = none',
            # 'pr = 0.4, d = Threshold, r = Mag',
            # 'pr = 0.6, d = Threshold, r = Non-zeros',
            # 'pr = 0.6, d = Threshold, r = none',
              ]
    # labels = None
    names = ['LEs/stacked_LSTM_full'] + ['LEs/stacked_LSTM_pruned'] * 3
    indices = [161, 354, 356, 370]
    # indices = [161] + [*range(340, 372, 1)]
    labels = []
    print(indices)

    sum = 0
    divider_indices = [0]
    for i, name in enumerate(names):
        print(i, name, indices[i])
        LE = LE_loading(folder_name=name, trial_index=indices[i], epochs=[1], num_epochs=40)
        sum += LE.shape[0]
        if name == 'LEs/stacked_LSTM_full':
            labels.append(['pr = 0'])
        else:
            labels.append([f'pr = 0.8 {indices[i]}'])
        # plt.figure(2)
        # colors = ['blue', 'green', 'purple']

        # for j in range(3):
        # # for j in range(LE.shape[0]):
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
    # print(divider_indices)
    tsne(LEs, dim=2, divider_indices=divider_indices, names=names, labels=labels, use_tsne=False, limited_samples=40)
    num_epochs = 100
    perplexity_compare(names, indices, num_epochs, labels=labels)
if __name__ == '__main__':
    main()