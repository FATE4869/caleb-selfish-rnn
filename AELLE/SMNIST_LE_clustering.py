import torch
from AEPredNet import AEPredNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
from SMNIST.util import *

def tsne(X, targets, dim=2):
    tsne_model = TSNE(perplexity=10, n_components=dim, random_state=1)

    x_embedded = tsne_model.fit_transform(X.detach().numpy())

    # x_embedded = tsne(hidden_outputs.detach(), dim=tsne_dim)
    indices = np.argsort(x_embedded[:, 0])
    a =x_embedded[indices, 0]
    b = x_embedded[indices, 1]
    fig = plt.figure()
    if (dim==2):
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=6)
        # plt.xlim([-60, 40])
        # plt.ylim([-60, 80])
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1],
                   x_embedded[:, 2], s=6)
    plt.legend()
    plt.title("TSNE")
    plt.show()
    print(x_embedded.shape)

    tsne_results = {'x_embedded':x_embedded, 'targets':targets}
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne.png', dpi=200)

    return tsne_results

def tsne_perf(dim, path, x_embedded=None):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p = ax.scatter(x_embedded[:, 0], x_embedded[:, 1], s=10, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(x_embedded[:, 0], x_embedded[:, 1],
                   x_embedded[:, 2], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        ax.set_xlabel('TSNE 1')
        ax.set_ylabel('TSNE 2')
        ax.set_zlabel('TSNE 3')
    plt.title("TSNE Performance")
    plt.show()
    # plt.savefig('../lyapunov-hyperopt-master/Figures/AEPredNet_tsne_performance.png', dpi=200)

def tsne_binary(dim, path, x_embedded=None, threshold=0.1):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    # targets_binary = np.ones_like(targets)
    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    if dim == 2:
        ax0 = plt.subplot(3, 2, 2)
        ax0.scatter(x_embedded[indices_bad, 0], x_embedded[indices_bad, 1],  s=15, c='r', label="Bad performace")
        ax0.scatter(x_embedded[indices_good, 0], x_embedded[indices_good, 1], c='g',s=6, label="Good performace")
        ax0.legend()
        ax0.set_title(threshold)
    elif dim==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_embedded[indices_bad, 0], x_embedded[indices_bad, 1], x_embedded[indices_bad, 2],
                    s=15, c='r', label="Bad performace")
        ax.scatter(x_embedded[indices_good, 0], x_embedded[indices_good, 1], x_embedded[indices_good, 2],
                    s=6, c='g', label="Good performace")
        plt.legend()
        plt.title(threshold)
    # plt.savefig('../lyapunov-hyperopt-master/Figures/tsne/binary/threshol_{}.png'.format(threshold), dpi=200)

    # plt.show()

def tsne_hist(dim, path, x_embedded=None, threshold=0.1, num_bins = 50 ):
    if not x_embedded:
        tsne_results = torch.load(path + 'tsne.p')
        x_embedded, targets = tsne_results['x_embedded'], tsne_results['targets']
    # targets_binary = np.ones_like(targets)
    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]

    ax1 = plt.subplot(3, 2, 4)
    # First Dimension
    good_points = x_embedded[indices_good, :]
    counts, bins = np.histogram(good_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='g')

    bad_points = x_embedded[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 0], bins= num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='r')
    ax1.set_ylim([0, 50])
    ax1.set_title("dim 0")

    # Second Dimension
    ax2 = plt.subplot(3, 2, 1)
    good_points = x_embedded[indices_good, :]
    counts, bins = np.histogram(good_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='g')

    bad_points = x_embedded[indices_bad, :]
    counts, bins = np.histogram(bad_points[:, 1], bins= num_bins)
    ax2.hist(bins[:-1], bins, weights=counts,orientation="horizontal", color='r')
    ax2.set_title("dim 1")
    ax2.set_xlim([0, 50])

    ax3 = plt.subplot(3, 2, 3)
    ax3.hist2d(good_points[:,0], good_points[:,1 ], bins=20)
    ax3.set_xlim([-80,80])
    ax3.set_ylim([-55, 55])
    ax3.set_title("Good performance")

    ax4 = plt.subplot(3, 2, 5)
    ax4.hist2d(bad_points[:,0], bad_points[:,1 ], bins=20)
    ax4.set_xlim([-80,80])
    ax4.set_ylim([-55, 55])
    ax4.set_title("Bad Performance")

    # plt.savefig('../lyapunov-hyperopt-master/Figures/tsne/combined/threshold_{}.png'.format(threshold), dpi=200)

    plt.show()

def pca(X, targets, dim=2, basics=None):

    U,S,V = torch.pca_lowrank(X)
    if basics is not None:
        V = basics
    low_rank = torch.matmul(X, V[:, :dim]).detach().numpy()
    # fig = plt.figure()
    # if (dim==2):
    #     plt.scatter(low_rank[:, 0], low_rank[:, 1], s=6)
    # elif (dim==3):
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(low_rank[:, 0], low_rank[:, 1],
    #                low_rank[:, 2], s=6)
    # plt.legend()
    # plt.title("PCA")
    # plt.show()
    pca_results = {"low_rank": low_rank, "targets": targets, "basics": V}
    return pca_results

def pca_perf(dim, path=None, low_rank=None, targets=None):
    if path is not None:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']
    fig = plt.figure()
    if (dim == 2):
        ax = fig.add_subplot(111)
        p = ax.scatter(low_rank[:, 0], low_rank[:, 1], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
    elif (dim==3):
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(low_rank[:, 0], low_rank[:, 1],
                   low_rank[:, 2], s=6, c = targets,
                       norm=colors.LogNorm(vmin=targets.min(), vmax=targets.max()))
        fig.colorbar(p, label='Val Loss')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
    plt.title("PCA Performance")
    plt.show()

def pca_binary(dim, path=None, low_rank=None, targets=None, threshold=0.1):
    if path is not None:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    print(len(indices_good), len(indices_bad))


    if (dim==2):
        # ax0 = plt.subplot(3, 2, 2)
        ax0 = plt.subplot(1,1,1)

        ax0.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], c='lime', s=100,alpha=0.5, label="Good performace")
        ax0.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1],  s=100, c='r', alpha=0.5, label="Bad performace")


        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['', '', '', '', '', ''])
        # plt.xticks([-3, -2, -1, 0, 1, 2, 3, 4], ['', '', '', '', '', '', '', ''])
        # plt.yticks([-3, -2, -1, 0, 1, 2, 3, 4], ['', '', '', '', '', '', '', ''])
        # ax0.set_title('')
        ax0.set_axis_off()
        # plt.xticks([-3, 3])
        # ax0.set_xlabel('PC 1')
        # ax0.set_ylabel('PC 2')
        # ax0.set_xlim([-1.5, 5.0])
        # ax0.set_ylim([-1.0, 2.0])
        # plt.legend()
        # ax0.set_title(threshold)
    elif (dim==3):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1], low_rank[indices_bad, 2],
                    s=15, c='r', label="Bad performace")
        ax.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], low_rank[indices_good, 2],
                    s=6, c='g', label="Good performace")
        ax.view_init(elev=-20, azim=-30)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        plt.legend()
        plt.title(threshold)
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/binary/threshol_{}.png'.format(threshold), dpi=200)
    # plt.show()

def pca_parameter(dim, path, low_rank=None, targets=None, indices=None):
    if not low_rank:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']
    if not indices:
        indices = [0, 100, 200, 300, 400, 500]
    num_colors = 5
    # threshold_range = [max(targets), .8, .6, .4, .2, 0]
    # threshold_range = [0, .1, .2, .4, .6, max(targets)]
    gradient = np.linspace(1, 0.5, num_colors)
    cmap = plt.cm.get_cmap('brg')

    plt.figure()
    for i in range(num_colors):
        plt.scatter(low_rank[indices[i]: indices[i + 1], 0], low_rank[indices[i]: indices[i + 1], 1],
                    c=cmap(gradient[i] * np.ones_like(low_rank[indices[i]: indices[i + 1], 1])))
    plt.show()


def pca_hist(dim, path=None, low_rank=None, targets=None, threshold=0.1, num_bins=50):
    if low_rank is None:
        pca_results = torch.load(path + 'PCA_dim{}.p'.format(dim))
        low_rank, targets = pca_results['low_rank'], pca_results['targets']

    bool_arr_bad = targets > threshold
    bool_arr_good = targets <= threshold
    indices_bad = np.where(bool_arr_bad)[0]
    indices_good = np.where(bool_arr_good)[0]
    threshold = 5
    num_good_samples_over_5 = len(np.where(low_rank[indices_good, 0] > threshold)[0])
    num_bad_samples_over_5 = len(np.where(low_rank[indices_bad, 0] > threshold)[0])
    print(f'Good sample over 5: {num_good_samples_over_5}/{len(indices_good)} = {num_good_samples_over_5/len(indices_good):.4f} \n'
          f'Bad samples over 5: {num_bad_samples_over_5}/{len(indices_bad)} = {num_bad_samples_over_5/len(indices_bad):.4f}')
    if num_good_samples_over_5 is not 0 or num_bad_samples_over_5 is not 0:
        precision = num_good_samples_over_5 / (num_good_samples_over_5 + num_bad_samples_over_5) * 100
        recall = num_good_samples_over_5 / len(indices_good) * 100
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'precision:{precision:.2f}%, recall: {recall:.2f}, f1 score: {f1_score:.2f}')
        # print(f'Recall: ')
    # print(f"There are {num_good_samples_over_5} good samples out of {len(indices_good)} have pc1 over 5, the "
    #       f"percentage is {num_good_samples_over_5 / len(indices_good):.4f}")
    ax0 = plt.subplot(2, 1, 1)

    ax0.scatter(low_rank[indices_bad, 0], low_rank[indices_bad, 1], s=100, c='r', alpha=0.4, label="Bad performace")
    ax0.scatter(low_rank[indices_good, 0], low_rank[indices_good, 1], c='lime', s=100, alpha=0.6,
                label="Good performace")
    ax0.set_xlim([-6, 8])
    ax0.set_axis_off()

    ax1 = plt.subplot(2, 1, 2)
    # First Dimension
    # good_points = low_rank[indices_good, :]
    counts, bins = np.histogram(low_rank[:, 0], bins=num_bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='lime')

    bad_points = low_rank[indices_bad, :]
    counts, bins_bad = np.histogram(bad_points[:, 0], bins=bins)
    ax1.hist(bins[:-1], bins, weights=counts, color='r')
    ax1.set_xlim([-6, 8])
    plt.xticks([-6, -4, -2, 0, 2, 4, 6, 8])
    ax1.set_ylim([0, 40])
    plt.yticks([0, 20, 40])
    # ax1.set_title("PC 1")
    # ax1.set_axis_off()
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/dim_0/threshol_{}.png'.format(threshold), dpi=200)
    # plt.show()
    # Second Dimension
    # ax2 = plt.subplot(2, 1, 2)
    # good_points = low_rank[indices_good, :]
    # counts, bins = np.histogram(good_points[:, 1], bins= num_bins)
    # ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='g')
    #
    # bad_points = low_rank[indices_bad, :]
    # counts, bins = np.histogram(bad_points[:, 1], bins= num_bins)
    # ax2.hist(bins[:-1], bins, weights=counts, orientation="horizontal", color='r')
    # ax2.set_xlim([0, 50])
    # ax2.set_title("PC 2")
    #
    # ax3 = plt.subplot(3, 2, 3)
    # ax3.hist2d(good_points[:,0], good_points[:,1 ], bins=20)
    # ax3.set_xlim([-8.5, 2.5])
    # ax3.set_ylim([0, 5.5])
    # ax3.set_title("Good performance")

    # ax4 = plt.subplot(3, 2, 5)
    # ax4.hist2d(bad_points[:,0], bad_points[:,1 ], bins=20)
    # ax4.set_xlim([-8.5, 2.5])
    # ax4.set_ylim([0, 5.5])
    # ax4.set_title("Bad Performance")
    # plt.savefig('../lyapunov-hyperopt-master/Figures/pca/combined/threshold_{}.png'.format(threshold), dpi=200)
    plt.show()

def visualization(targets, predictions):
    plt.figure()
    plt.scatter(torch.ones_like(targets) , targets)
    plt.scatter(torch.ones_like(predictions) * 1.1, predictions.detach())
    plt.legend(["Targets", 'Predictions'])
    # plt.axis([0.95, 1.15, -.1, 1.])
    plt.show()

def main():
    N_s = [64]
    # a_s = [1.0, 2.0, 5.0]
    a_s = 'random'
    # inputs_epochs = [1]
    inputs_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_epoch = 20
    val_split = 0.6
    prediction_loss_type = "L1"

    dim = 2
    # if inputs_epoch is None:
    results_path, _ = result_path_name(N_s, a_s, inputs_epochs)
    print(results_path+'{}/ae_prednet_4000.ckpt'.format(prediction_loss_type))

    model = torch.load(results_path+'{}/ae_prednet_6000.ckpt'.format(prediction_loss_type))
    model.load_state_dict(model.best_state)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # # validation
    # val_N_s = [64]
    # val_a_s = 'random'
    # inputs_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # test_inputs_epochs = [20]
    # inputs = torch.tensor([])
    # targets = torch.tensor([])
    # #
    # for N in val_N_s:
    #     if val_a_s == 'random':
    #         for inputs_epoch in test_inputs_epochs:
    #             data = pickle.load(open(f'../SMNIST/TrainingData/interpreted/N_{N}/a_{val_a_s}/train/epoch_{inputs_epoch}.pickle', 'rb'))
    #             inputs = torch.cat((inputs, data['inputs']), dim=0)
    #             targets = torch.cat((targets, data['targets']), dim=0)
    #     else:
    #         for a in val_a_s:
    #             for inputs_epoch in test_inputs_epochs:
    #                 data = pickle.load(
    #                     open(f'../SMNIST/TrainingData/interpreted/N_{N}/a_{a}/test/epoch_{inputs_epoch}.pickle', 'rb'))
    #                 inputs = torch.cat((inputs, data['inputs']), dim=0)
    #                 targets = torch.cat((targets, data['targets']), dim=0)
    # print(inputs.shape, targets.shape)


    # print(Results_path+ 'epochs.pickle')
    if len(inputs_epochs) > 1:
        split = pickle.load(open(results_path + 'epochs.pickle', 'rb'))
    else:
        split = pickle.load(open(results_path + f'epoch_{inputs_epochs[0]}.pickle', 'rb'))
    # inputs, targets = split['train_data'], split['train_targets']
    inputs, targets = split['val_data'], split['val_targets']
    # print(inputs, targets)
    # inputs = torch.cat([inputs, split['val_data']])
    # targets = torch.cat([targets, split['val_targets']])

    # print(torch.max(inputs), torch.min(inputs))
    # print(torch.max(targets), torch.min(targets))
    rec_outputs, hidden_outputs, pred_outputs = model(inputs)
    visualization(targets, pred_outputs)
    # print(torch.mean(abs(targets - pred_outputs)))

    # pca
    for i in range(10, 11):
        threshold = 0.5
    #     # threshold = (i + 1) * 0.01
    #
        basics = torch.load(results_path+f'/{prediction_loss_type}/PCA_basics_dim{dim}.p')
        print(basics.shape)
        pca_results = pca(X=hidden_outputs, dim=dim, targets=targets, basics=basics)
    #
    #     # pca_results = pca(X=hidden_outputs, dim=dim, targets=targets, basics=None)
    #     # basics = pca_results['basics']
    #     # torch.save(basics, results_path+f'/{prediction_loss_type}/PCA_basics_dim{dim}.p')
    #
    #
    #     # pca_perf(dim=dim, path=Results_path + prediction_loss_type + '/')
    #     # pca_parameter(dim=dim, path=Results_path + prediction_loss_type + '/')
    #     # pca_binary(dim, low_rank=pca_results['low_rank'], targets=targets, threshold=threshold)
    #     # pca_binary(dim, low_rank=pca_results['low_rank'], targets=pca_results['targets'], threshold=threshold)
    #     # plt.show()
        pca_hist(dim, low_rank=pca_results['low_rank'], targets=targets, threshold=threshold)



if __name__ == "__main__":
    main()
