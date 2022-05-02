import matplotlib.pyplot as plt
import pickle
import numpy as np

import statistics
from math import sqrt
import scipy.stats

def plot_confidence_interval(x, values, confidence=0.95, color='#2187bb', horizontal_line_width=0.15):
    n = len(values)
    m, se = np.mean(values), scipy.stats.sem(values)
    confidence_interval = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    left = x - horizontal_line_width / 2
    top = m - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = m + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, m, 'o', color='#f44336')

    return m, confidence_interval

def main():
    losses = []
    for i in range(5):
        trials = pickle.load(open(f'../trials/trials_{i+1}.pickle', 'rb'))
        losses.append(trials.best_trial['result']['loss'])
    print(losses)
    fig, ax = plt.subplots()
    ax.scatter(np.ones_like(losses), losses, label='tpe')

    plot_confidence_interval(1, losses)

    plt.scatter([2], 75.68, label='loss curve')
    plt.scatter([0], 72.46, label='LE distance')
    plt.xticks([])
    plt.ylabel('ppl')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()