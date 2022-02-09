import torch
from AE_utils import mini_batch_ae, train_val_split
import os
import util
from util import *
from preprocess import generateDataset
from SMNIST_AEPredNet import AEPredNet
from SMNIST_AC_train import ae_train



def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # hyper parameters
    folder_paths = [
                    '../../../dataset/PTB/LEs/stacked_LSTM_full/',
                    '../../../dataset/PTB/LEs/stacked_LSTM_pruned/'
                    ]
    indices = [
        141, 163, 165, 167, 169, 171, 173, 175
    ]
    pruning_rates = [
        0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.85, 0.95
    ]
    dataset = generateDataset(folder_paths=folder_paths, indices=indices,
                    pruning_rates=pruning_rates)
    inputs = dataset['LEs']
    targets = dataset['targets']
    N_s = [3000]
    a_s = [0.1]
    inputs_epochs = [0, 1, 2]
    target_epoch = 30
    # inputs, targets = util.train_data_loading(N_s, a_s, inputs_epochs, interpolated=True, isTrain=True)
    # targets_var = torch.sqrt(torch.var(targets))
    # print("targets_var:", targets_var)
    # targets /= targets_var

    # inputs = normalization(inputs)
    # targets = normalization(targets)
    print(f'inputs shape: {inputs.shape}, targets shape: {targets.shape}')
    print(f'max inputs: {torch.max(inputs)}, min inputs: {torch.min(inputs)}, variance inputs: {torch.var(inputs)}')
    print(f'max targets: {torch.max(targets)}, min targets: {torch.min(targets)}, variance targets: {torch.var(targets)}')
    val_split = 0.2
    prediction_loss_type = "L1"
    split = train_val_split(data=inputs, targets=targets, val_split=val_split)

    result_folder, result_file = result_path_name(N_s, a_s, inputs_epochs=inputs_epochs, target_epoch=target_epoch)
    print(f'result_folder: {result_folder}')
    print(f'result_file: {result_file}')
    data_saving(split, folder_name=result_folder, file_name=result_file)

    plt.figure()
    plt.scatter(torch.zeros_like(targets), targets)
    plt.show()
    x_train, y_train, x_val, y_val = split['train_data'], split['train_targets'],\
                                     split['val_data'], split['val_targets']
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    model = AEPredNet(input_size=3000, latent_size=128, lr=1e-4, act='tanh', device=device, prediction_loss_type=prediction_loss_type)
    # alphas = [10, 10, 20, 20, 30, 30]
    alphas = [5, 5, 10, 20]
    for i, alpha in enumerate(alphas):
        model = ae_train(model, x_train, y_train, x_val, y_val, alpha=alpha, epochs=1000,
                         print_interval=500, batch_size=128, val_batch_size=128)
        model_saving(model, N_s, a_s, inputs_epochs)
        model.lr = model.lr / 4
    plt.figure()
    plt.plot(range(model.global_step), model.train_loss, label='train')
    plt.plot(range(model.global_step), model.val_loss, label='total')
    plt.plot(range(model.global_step), model.vl1, label='rec loss (L1)')
    plt.plot(range(model.global_step), model.vl2, label='pred loss (L1)')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.show()
if __name__ == "__main__":
    main()
