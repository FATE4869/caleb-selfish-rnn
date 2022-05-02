import pickle
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from sources.train import train_main
from sources.args import Args
import math
from LE_calculation import *
from util import *

count = 11000

def main(num_epochs = 2, max_evals = 2, LE_based=False):
    space = {
        "sparse_init": hp.choice("sparse_init", ['uniform', 'ER']),
        "growth": hp.choice("growth", ['random']),
        "death": hp.choice("death", ['magnitude', 'SET', 'threshold', 'global_magnitude']),
        "redistribution": hp.choice("redistribution", ['magnitude', 'nonzeros', 'none']),
        # "death_rate": hp.randint('death_rate', 6), # Returns a random integer in the range [0, upper)
        # "lr": hp.randint('lr', 5) # Returns a random integer in the range [0, upper)
    }
    trials = Trials()

    # define an objective function
    def objective(params):
        args = Args().args
        args.epochs = num_epochs
        args.eval_batch_size = 20
        global count
        args.save = f'{count}.pt'
        args.init = params['sparse_init']
        args.growth = params['growth']
        args.death = params['death']
        args.redistribution = params['redistribution']
        # args.death_rate = 0.1 * (params['death_rate'] + 4)
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
    if not os.path.exists(trials_path):
        os.mkdir(trials_path)
    if LE_based:
        pickle.dump(trials, open(f'{trials_path}/LE_tpe_trials_num_{max_evals}.pickle', 'wb'))
    else:
        pickle.dump(trials, open(f'{trials_path}/PPL_tpe_trials_num_{max_evals}.pickle', 'wb'))
    return trials

if __name__ == '__main__':
    num_epochs = 3
    max_evals = 20
    LE_based = False
    trials = main(num_epochs, max_evals, LE_based)