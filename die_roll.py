"""
Simulation of rolling a 6 sided die.

# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
# https://github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing

https://towardsdatascience.com/calogica-com-dice-polls-dirichlet-multinomials-eca987e6ec3f
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
from scipy.stats import beta
import random

NUM_TRIALS = 20
# loaded_on_6
die1 = np.array([0.05, 0.15, 0.2, 0.15, 0.15, 0.3])
# fair die
die2 = np.ones(6) / 6
# loaded_on_4
die3 = np.array([0.05, 0.25, 0.2, 0.4, 0.1, 0.1])


class Die:
    # we first sample, and highest sample is most likely to be best result,
    # and we update after each pull

    def __init__(self, weights, name):
        self.weights = np.array(weights) # sums to one, will be the prob of each die roll
        self.alphas = np.ones(6)
        self.name = name
        # over a number of rolls, we should see the updated alphas converge to look like
        # the weights of this particular die

    def sample(self):
        return dirichlet.rvs(self.alphas)[0]

    def roll(self):
        """Return index 0-5 of category (number) chosen/rolled.

        Parameters
        ----------
        None

        Returns
        -------
        roll : an integer 0-5 that indicates which category is chosenrollcount
            -- flattened is a list of 1000 items consisting of 0 repeated
            1000*weight_0 times, and so on all the categories, so that a
            random choice from flattened will be a choice with specified
            probability

        Example
        -------
        for self.weights= [.01, .04, .15, 0.3, 0.1, 0.4],
        flattened would be [0, 1, 1, 1, 1, 2 ...etc]
        where 0 appears 1x, 1 appears 4x, 2 15x, 5 40x etc.
        """
        rolls = []

        for i in range(len(self.weights)):
            rolls.append([i] * int(self.weights[i] * 1000))

        flattened = []
        for sublist in rolls:
            for val in sublist:
                flattened.append(val)
        roll = random.choice(flattened)

        return roll

    def update(self, roll):
        # roll is an integer between 0 and 5, so we increment by one the alpha
        # corresponding to the number rolled, since 1 is the reward
        self.alphas[roll] = self.alphas[roll] + 1


def print_dice(dice):
    print('\nDice: Total rolls:', dice[0].alphas.sum()
          + dice[1].alphas.sum() + dice[2].alphas.sum() - 18)
    print('----------------------------------------------')
    for d in dice:
        print(d.name, ': Rolls = ', d.alphas.sum() - 6, '\n alphas:',
              d.alphas, '\n weights:   ', np.around(d.weights, 3))
        print(' alpha_ratio', np.around(d.alphas/d.alphas.sum(), 3), '\n')


def plot(dice, trial, samples):
    x = np.linspace(0, 1, 200)
    print('----------------------------------------------')
    print('\nTotal    : Rolls = ', dice[0].alphas.sum() + dice[1].alphas.sum()
          + dice[2].alphas.sum() - 18)
    for d in dice:
        print(d.name, ': Rolls = ', d.alphas.sum()-6)

    for d in dice:
        # the distribution for each number is a Beta, where a-1= # times 'number'
        # appeared for the given die up to the trial
        for number in range(6):
            p = round(d.weights[number], 3)
            a = int(d.alphas[number] - 0)
            # initialized alphas is 1 for all numbers
            b = int(d.alphas.sum() - d.alphas[number])
            # N is # rolls of the die
            N = d.alphas.sum()-6
            p_est = a / (a + b)
            y = beta.pdf(x, a, b)
            lab = 'N={:0.0f}  real p={:0.3f}   p_est={:0.3f} a={:d}  b={:d}'.format(N, p, p_est, a, b)
            plt.plot(x, y, label=lab)
        thetitle = "%s Die: Distributions after %s trials" % (d.name, trial)
        plt.title(thetitle)
        plt.legend()
        plt.axvline(x=.1667, ymin=0, ymax=1)
        plt.show()


def experiment():
    dice = [Die(die1, '6_loaded'), Die(die2, 'Fair'), Die(die3, '4_loaded')]

    sample_points = [5, 10, 20, 50, 100, 500, 1000, 2000]
    for i in range(NUM_TRIALS):
        loaded_die = None
        maxnumber = -1
        maxloading = 0
        samples = []

        for d in dice:
            test = d.sample()
            samples.append(test)
            for j in range(test.size):
                if test[j] > maxloading:
                    maxloading = test[j]
                    maxnumber = j + 1
                    loaded_die = d
        loaded_die.update(loaded_die.roll())

        if i+1 in sample_points:
            print('Trial', i+1)
            plot(dice, i+1, samples)

    return dice


if __name__ == '__main__':
    dice = experiment()
    print_dice(dice)
