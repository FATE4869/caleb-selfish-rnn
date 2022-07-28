import pickle
import os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL, atpe
from train import train_main
from args import Args
import math
from LE_calculation import *
from util import *

count = 18000

def main(args):
    parser = argparse.ArgumentParser(description="Train recurrent models")
    parser.add_argument("-epochs", type=int, default=3, required=False)
    parser.add_argument("-no_cand", type=int, default=2, required=False, help='number of candidates')
    parser.add_argument("-LE_based", type=bool, default=True, required=False,
                        help='Use LE_based metric if true, else use ppl metric')
    parser.add_argument("-count", type=int, default=10000, required=False, help='index of trial')
    parser.add_argument("-sparsity", type=float, default=0.0, required=False)
    parser.add_argument("-fix", type=bool, default=False, required=False,
                        help='whether to fix the pruning mask over training')
    parser.add_argument("-sparse_init", type=str, default='uniform', required=False, choices=['uniform', 'ER'])
    parser.add_argument("-growth", type=str, default='random', required=False, choices=['random'])
    parser.add_argument("-death", type=str, default='magnitude', required=False,
                        choices=['magnitude', 'SET', 'threshold', 'global_magnitude'])
    parser.add_argument("-redistribution", type=str, default='magnitude', required=False,
                        choices=['magnitude', 'nonzeros', 'none'])
    parser.add_argument("-death_rate", type=float, default=0.6, required=False)
    parser.add_argument("-decay_schedule", type=str, default='cosine', required=False)
    parser.add_argument("-seed", type=int, default=1111, required=False)

    args.update_frequency = 100
    args.decay_schedule = 'cosine'
    args.dense_allocation = None
    args.density = 1 - args.sparsity
    args.seed = 1111




    args = Args().args
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # dataset
    args.data = '/home/ws8/caleb/dataset/PTB/penn/'  # location of the data corpus
    args.batch_size = 20  # batch size
    args.eval_batch_size = 20  # evaluate batch size
    args.bptt = 35  # sequence length

    # model hyper-parameters
    args.model = 'LSTM'  # type of recurrent net (RHN, LSTM)
    args.emsize = 1500  # size of word embeddings
    args.nhid = 1500  # number of hidden units per layer
    args.nlayers = 2  # number of layers
    args.dropout = 0.65  # dropout applied to layers (0 = no dropout)
    args.dropouth = 0.25  # dropout for rnn hidden units (0 = no dropout)
    args.dropouti = 0.65  # dropout for input embedding layers (0 = no dropout)
    args.dropoute = 0.2  # dropout to remove words from embedding layer (0 = no dropout)
    args.tied = False
    args.couple = True

    # optimizer
    args.lr = 40  # initial learning rate
    args.clip = 0.25  # gradient clipping
    args.optimizer = 'sgd'
    args.momentum = 0.9  # SGD momentum (default: 0.9)
    args.wdecay = 1.2e-6

    args.log_interval = 200
    args.evaluate = ''  # path to pre-trained model (default: none)
    randomhash = ''.join(str(time.time()).split('.'))
    args.save = randomhash + '.pt'
    args.sparse = True
    args.seed = 1111
    args.log_interval = 200
    args.keep_train_from_path = None

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
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
