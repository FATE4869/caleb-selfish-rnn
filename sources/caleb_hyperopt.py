import pickle
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, atpe
from train import train_main
from args import Args
import math
from LE_calculation import *
from util import *

count = 18000

def main(num_epochs = 2, max_evals = 2, LE_based=False):
    space = {
        "sparse_init": hp.choice("sparse_init", ['uniform', 'ER']),
        "growth": hp.choice("growth", ['random']),
        "death": hp.choice("death", ['magnitude', 'SET', 'threshold', 'global_magnitude']),
        "redistribution": hp.choice("redistribution", ['magnitude', 'nonzeros', 'none']),
        # "death_rate": hp.randint('death_rate', 500),  # Returns a random integer in the range [0, upper)
        "death_rate": hp.randint('death_rate', 6), # Returns a random integer in the range [0, upper)
        # "lr": hp.randint('lr', 5) # Returns a random integer in the range [0, upper)
    }
    trials = Trials()

    # define an objective function
    def objective(params):
        args = Args().args
        args.sparsity = 0.67
        args.density = 1 - args.sparsity
        args.epochs = num_epochs
        args.eval_batch_size = 20
        args.seed = 1111
        global count
        count_local = count
        args.save = f'{count}.pt'
        args.init = params['sparse_init']
        args.growth = params['growth']
        args.death = params['death']
        args.redistribution = params['redistribution']
        # args.death_rate = 0.001 * (params['death_rate'] + 400)
        args.death_rate = 0.1 * (params['death_rate'] + 4)
        # args.lr = 10 * (params['lr'] + 1)
        args.verbose = False
        val_loss = train_main(args, 'cuda')
        if not LE_based:
            count += 1
            return {"loss": math.exp(val_loss), "status": STATUS_OK, 'args': args}
        else:
            args.trial_num = count
            args.eval_batch_size = 2
            LE_main(args)

            LE_distance = LE_distance_main(count, num_epochs=num_epochs)
            print(f"count: {count} \t LE_distance: {LE_distance}")
            count += 1
            return {"loss": LE_distance, "status": STATUS_OK, 'val_loss': math.exp(val_loss), 'args': args}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    print(best)
    print(trials)
    trials_path = '../trials/'
    count_local = 18000
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)
    if LE_based:
        pickle.dump(trials, open(f'{trials_path}/LE_tpe_trials_num_{max_evals}_ind_{count_local}.pickle', 'wb'))
    else:
        pickle.dump(trials, open(f'{trials_path}/PPL_tpe_trials_num_{max_evals}_ind_{count_local}.pickle', 'wb'))
    return trials

if __name__ == '__main__':
    num_epochs = 3
    max_evals = 2
    LE_based = True
    trials = main(num_epochs, max_evals, LE_based)
    # trials = pickle.load(open('../trials/LE_tpe_trials_num_20_ind_15000.pickle', 'rb'))
    LE_distances = []
    val_losses = []
    test_losses = []
    for i in range(max_evals):
        LE_distances.append(round(trials.results[i]['loss'], 2))
        val_losses.append(round(trials.results[i]['val_loss'], 2))
        test_losses.append(round(trials.results[i]['test_loss'], 2))
    print(f"LE_distances: {LE_distances}")
    print(f"val_losses: {val_losses}")
    print(f"test_losses: {test_losses}")
