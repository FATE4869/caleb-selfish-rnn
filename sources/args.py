import argparse
import os
import time
import torch

def ed(param_name, default=None):
    return os.environ.get(param_name, default)

class Args():
    def __init__(self):

        parser = argparse.ArgumentParser(description='Network Pruning')
        args = parser.parse_args()
        # general hyper-parameters
        use_cuda = torch.cuda.is_available()
        args.device = torch.device("cuda" if use_cuda else "cpu")

        # dataset
        args.data = '/home/ws8/caleb/dataset/PTB/penn/'  # location of the data corpus
        args.epochs = 100  # upper epoch limit
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

        args.log_interval = 200
        args.evaluate = ''  # path to pre-trained model (default: none)
        randomhash = ''.join(str(time.time()).split('.'))
        args.save = randomhash + '.pt'
        args.sparse = True
        args.seed = 1111
        args.log_interval = 200
        args.keep_train_from_path = None
        self.args = args
        self.opt()

        self.sparse()
        # self.rigL()
        self.selfish_rnn()
    def opt(self):
        self.args.lr = 40  # initial learning rate
        self.args.clip = 0.25  # gradient clipping
        self.args.optimizer = 'sgd'
        self.args.momentum = 0.9  # SGD momentum (default: 0.9)
        self.args.wdecay = 1.2e-6

    def selfish_rnn(self):
        self.args.beta = 1  # beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
        self.args.nonmono = 5  # random seed ?
        self.args.death_rate = 0.8
    # parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    # parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    # parser.add_argument('--sparse_init', type=str, default='uniform', help='sparse initialization')
    # parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, gradient.')
    # parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    # parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    # parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    # parser.add_argument('--density', type=float, default=0.33, help='The density of the overall sparse network.')
    # parser.add_argument('--update_frequency', type=int, default=100, metavar='N',
    #                     help='how many iterations to train between parameter exploration')
    # parser.add_argument('--decay-schedule', type=str, default='cosine',
    #                     help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')

    def sparse(self):

        self.args.fix = False
        self.args.sparsity = 0.66
        self.args.density = 0.33
        self.args.sparse_init = 'uniform'
        self.args.death = 'magnitude'
        # self.args.death_rate = death_rate
        self.args.growth = 'random'
        self.args.redistribution = 'none'
        self.args.update_frequency = 100
        self.args.decay_schedule = 'cosine'
        self.args.dense_allocation = None

        # RigL
    def rigL(self):
        self.args.sparsity = 0.1
        self.args.delta = 100
        self.args.alpha = 0.3
        self.args.static_topo = False # if 1, use random sparisty topo and remain static, else 0
        self.args.grad_accumulation_n = 1



def main():
    args = Args().args
    print(args)
if __name__ == '__main__':
    main()