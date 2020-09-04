"""
@author: pjrowe2012.

# For the course: Bayesian Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
# https://github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing

References
----------
DONE 8/30 - An Empirical Evaluation of Thompson Sampling 9 pg
        papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf

DONE - The Unbiased Estimate of the Covariance Matrix
        lazyprogrammer.me/covariance-matrix-divide-by-n-or-n-1/

REVIEWED 9/1 - Finite-time Analysis of the Multiarmed Bandit Problem  22 pg
        link.springer.com/article/10.1023/A:1013689704352

DONE 9/1 - Analysis of Thompson Sampling for the Multi-armed Bandit
        Problem 26 pg
        proceedings.mlr.press/v23/agrawal12/agrawal12.pdf
DONE 9/1 - Algorithms for the multi-armed bandit problem 32 pg
        www.cs.mcgill.ca/~vkules/bandits.pdf

REVIEWED 9/1- UCB REVISITED: IMPROVED REGRET BOUNDS FOR THE STOCHASTIC
        MULTI-ARMED BANDIT PROBLEM 11 pg
        personal.unileoben.ac.at/rortner/Pubs/UCBRev.pdf

TO DO - A Tutorial on Thompson Sampling 96pg
        web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


NUM_TRIALS = 10000
PROBABILITIES = [0.2, 0.5, 0.75]
# ep = epsilon, the % of time we randomly explore one of the bandits
# 1- ep = probability we just choose the Bandit with the max
ep = 0.1


class Bandit:

    def __init__(self, p):
        # p is win rate; p_estimate is our estimate for win rate, which is
        # updated with every pull of the bandit
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.pulls = [] # our record of all the pulls, list of 1s and 0s

    def pull(self):
        # return an int 1 or 0 instead of a boolean
        return 0 + (np.random.random() < self.p)

    def update(self, x):
        self.pulls.append(x)
        self.N += 1
        self.p_estimate = (1 - 1/self.N) * self.p_estimate + 1/self.N * x
        # print('After ',self.N, 'roll, new p_estimate=',self.p_estimate)
        # print('Samples: ', self.pulls,'\n')


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        lab = 'real p = {:0.4f}; a = {:0.4f} b = {:0.4f}'.format(b.p, b.a, b.b)
        plt.plot(x, y, label=lab)
    plt.title("Distributions after %s trials" % trial)
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in PROBABILITIES]
    # sample_points = [5, 10, 20, 50, 100, 500, 1000, 1999]
    rewards = np.zeros(NUM_TRIALS)
    num_explored = 0
    num_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])

    for i in range(NUM_TRIALS):
        # epsilon greedy to select next bandit
        if np.random.random() < ep:
            num_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if optimal_j == j:
            num_optimal += 1

        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)

    mean_means = sum(PROBABILITIES) / len(PROBABILITIES)
    expected_reward = NUM_TRIALS * ((1-ep) * max(PROBABILITIES)
                                    + ep * mean_means)
    max_reward = round(np.max(PROBABILITIES) * NUM_TRIALS, 2)
    print('Optimal bandit = ', optimal_j)
    print('Actual Means for bandits', PROBABILITIES)
    print('Mean estimates for bandits',
          [round(b.p_estimate, 3) for b in bandits])
    print('Mean of means: ', round(mean_means, 3))

    print('Number of trials:', NUM_TRIALS)
    print('Max rewards:', max_reward)
    print('Expected rewards:', round(expected_reward, 1))
    print('Total rewards:', rewards.sum())
    print('% max:', round(rewards.sum() / max_reward, 3))
    print('% expected:', round(rewards.sum() / expected_reward, 4))
    print('# times explored:', num_explored)
    print('# times exploited:', num_exploited)
    print('# times selected optimal bandit:', num_optimal)
    print('% selected optimal bandit:', num_optimal / NUM_TRIALS)

    cum_rewards = np.cumsum(rewards)
    win_rates = cum_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(PROBABILITIES))
    plt.ylim(ymin=0.4)
    plt.show()


if __name__ == '__main__':
    experiment()
