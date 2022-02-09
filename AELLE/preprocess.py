import os
import pickle
import torch
import numpy

def loadingLE(folder_path, index, last_epoch=30):
    count = 0
    for i in range(last_epoch):
        file_path = f'{folder_path}___e{i}___{index}.pickle'
        if os.path.exists(file_path):
            count += 1

            result = pickle.load(open(file_path, 'rb'))
            LE = torch.unsqueeze(result['LEs'], 1)
            inf_indices = torch.where(torch.isinf(LE))
            LE[inf_indices] = 0

            if i == 0:
                LEs = torch.unsqueeze(result['LEs'], 1)
            else:
                LEs = torch.cat([LEs, torch.unsqueeze(result['LEs'], 1)], dim=1)
    print(f'count: {count}')
    return torch.transpose(LEs, 1, 0)

def generateTarget(LE, isNegative):
    if isNegative:
        return torch.zeros(LE.shape[0])
    else:
        return torch.ones(LE.shape[0])

def generateDataset(folder_paths, indices, pruning_rates):
    for i in range(len(indices)):
        if i == 0: # only the first index is full network which is in a different folder
            folder_path = folder_paths[0]
        else:
            folder_path = folder_paths[1]
        for j in range(2): # each pruning rate repeats two times
            index = indices[i] + j

            LE = loadingLE(folder_path, index, last_epoch=30)
            isNegative= (pruning_rates[i] > 0.6)
            target = generateTarget(LE, isNegative)
            if i == 0 and j == 0:
                LEs = LE
                targets = target
            else:
                LEs = torch.cat([LEs, LE], dim=0)
                targets = torch.cat([targets, target], dim=0)
    print(LEs.shape)
    print(targets.shape)
    return {'LEs': LEs, 'targets': targets}
def main():
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


if __name__ == '__main__':
    main()