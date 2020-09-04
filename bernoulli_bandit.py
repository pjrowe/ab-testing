"""
@author: pjrowe2012.

This file runs an experiment with 3 bandits with binary outcome and different
means, plotting the Beta probability distribution of those means over time as
the experiment runs.

# From the course: Bayesin Machine Learning in Python: A/B Testing

References
----------
https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
https://github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

NUM_TRIALS = 2000
PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:

    def __init__(self, p):
        # Initialize a,b as 1, which is a uniform distribution [0,1] for the
        # mean of the bandit
        # p is the actual mean, but is unknown to the gambler
        self.p = p
        self.a = 1
        self.b = 1

    def sample(self):
        return np.random.beta(self.a, self.b)

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        # if a pull yields a 1 (success), then the 'a' is increased;
        # if pull yields 0, then 'b' increases by 1-0=1, and 'a' is increased
        # by 0
        self.a += x
        self.b += 1 - x


def plot(bandits, trial):
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        lab = 'N ={:0.0f} real p={:0.3f}; a={:0.0f} b={:0.0f}'.format(b.a + b.b - 2, b.p, b.a, b.b)
        plt.plot(x, y, label=lab)
    plt.title("Distributions after %s trials" % trial)
    plt.legend()
    plt.show()


def experiment():
    bandits = [Bandit(p) for p in PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 500, 1000, 1999]
    for i in range(NUM_TRIALS):
        bestb = None
        maxsample = -1
        allsamples = []
        for ba in bandits:
            sample = ba.sample()
            allsamples.append("%.4f" % sample)
            if sample > maxsample:
                maxsample = sample
                bestb = ba
        x = bestb.pull()
        bestb.update(x)

        if i+1 in sample_points:
            print("current samples: %s" % allsamples)
            plot(bandits, i+1)


if __name__ == '__main__':
    experiment()
