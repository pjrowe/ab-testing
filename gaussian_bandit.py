"""
@author: pjrowe2012.

# From the course: Bayesian Machine Learning in Python: A/B Testing
# deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# www.udemy.com/bayesian-machine-learning-in-python-ab-testing
# github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing

/towardsdatascience.com/calogica-com-dice-polls-dirichlet-multinomials-eca987e6ec3f

This file deals with simulating pulls of a Bandit with Gaussian reward.

The __main__ runs a number of simulations to see how the difference in mean
affects the rate of choosing the optimal bandit.  See scientific papers
mentioned in course syllabus to see more detailed discussion of theory of
upper bounds and lower bounds of 'regret'.

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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

NUM_TRIALS = 100


class Bandit:
    """Unknown mean, i.e., we initialize at 0; variance = 1."""

    def __init__(self, mean, name):
        self.mean = mean
        self.variance = 1
        self.tau = 1/self.variance

        self.predicted_mean = 0
        self.var0 = 1
        self.lam = 1 / self.var0
        self.name = name
        self.rolls = 1
        self.sum_of_x = 0


    def sample(self):
        """Sample from prior distribution.

        prior distribution's parameters are lam and predicted_mean
        """
        sample = np.random.normal()/np.sqrt(self.lam) + self.predicted_mean
        return round(sample, 3)


    def roll(self):
        """Sample the actual (unknown) distribution.

        Parameters are mean and tau, which are unknown to the gambler
        """
        roll = np.random.normal()/np.sqrt(self.tau) + self.mean
        return round(roll, 3)


    def update(self, x, debug=False):

        if debug:
            # print('\nsum_x before', self.sum_of_x)
            # print('x = ',x)
            self.sum_of_x += x
            # print('sum_x after', self.sum_of_x,'\n')
            print(self.name, 'mean0 before', self.predicted_mean)
            numerator = self.predicted_mean * self.lam + self.tau * x
            denominator = self.lam + 1 * self.tau
            self.predicted_mean = round(numerator / denominator, 3)
            print(self.name, 'mean0 after', self.predicted_mean, '\n')
            # print('lam before', self.lam)
            self.lam = self.lam + 1*self.tau
            # print('lam after',self.lam)

            print('------------------')
            # print('rolls before',self.rolls)
            self.rolls += 1
            # print('rolls after',self.rolls)

        else:
            self.sum_of_x += x
            numerator = self.predicted_mean * self.lam + self.tau * x
            denominator = self.lam + 1 * self.tau
            self.predicted_mean = round(numerator / denominator, 3)
            self.lam = self.lam + 1 * self.tau
            self.rolls += 1


def print_dice(dice):
    total_rolls = dice[0].rolls + dice[1].rolls + dice[2].rolls - 3
    print('\nDice: Total rolls:', total_rolls)
    print('----------------------------------------------')
    for d in dice:
        print(d.name, ': Rolls = ', d.rolls-1, '| Predicted Mean ',
              round(d.predicted_mean, 3), '| lambda', int(d.lam))


def plot(dice, trial, samples):
    x = np.linspace(-1, 8, 200)
    print('----------------------------------------------')
    total_rolls = dice[0].rolls + dice[1].rolls + dice[2].rolls - 3
    print('Total    : Rolls = ', total_rolls)
    for d in dice:
        print(d.name, ': Rolls = ', d.rolls - 1)
    print(samples)

    d1 = dice[0]
    d2 = dice[1]
    d3 = dice[2]
    mu1 = d1.mean
    mu2 = d2.mean
    mu3 = d3.mean
    y1 = norm.pdf(x, d1.predicted_mean, 1/(d1.lam**2))
    y2 = norm.pdf(x, d2.predicted_mean, 1/(d2.lam**2))
    y3 = norm.pdf(x, d3.predicted_mean, 1/(d3.lam**2))

    lab1 = 'real mu1 = {:0.3f} mu1_est = {:0.3f}  lam1_est = {:0.0f}'
    lab1 = lab1.format(mu1, d1.predicted_mean, d1.lam)
    lab2 = 'real mu2 = {:0.3f} mu2_est = {:0.3f}  lam2_est = {:0.0f}'
    lab2 = lab2.format(mu2, d2.predicted_mean, d2.lam)
    lab3 = 'real mu3 = {:0.3f} mu3_est = {:0.3f}  lam3_est = {:0.0f}'
    lab3 = lab3.format(mu3, d3.predicted_mean, d3.lam)

    plt.plot(x, y1, label=lab1)
    plt.plot(x, y2, label=lab2)
    plt.plot(x, y3, label=lab3)
    thetitle = "Distributions after %s trials" % (trial)
    plt.title(thetitle)
    plt.legend()
    plt.show()


def die_converge(mean, n):
    """Plot how predicted_mean of die converges to its actual mean.

    Parameters
    ----------
        mean: integer
            actual mean of die
        n:integer
            # of trials to plot

    Results / Examples
    ----------
        We see empirically that 50-100 trials is sufficient for the
        predicted_mean to fall within 5% of actual mean, and it remains
        within the band, IF the mean is set to 5 (>> variance of 1)

        For mean of 1-3,the results are less stable; more trials are required
        to converge, since variance is a larger in relation to the mean
    """
    die = Bandit(mean, 'name')
    means = []
    for i in range(n):
        means.append(die.predicted_mean)
        die.update(die.roll(), debug=False)
    x = np.linspace(1, n, n)
    plt.axhline(y=mean, xmin=0, xmax=n)
    plt.axhline(y=1.05*mean, xmin=0, xmax=n, ls='--')
    plt.axhline(y=0.95*mean, xmin=0, xmax=n, ls='--')
    plt.axvline(x=400, ymin=0, ymax=1, ls='--')
    plt.axvline(x=200, ymin=0, ymax=1, ls='--')
    plt.axvline(x=100, ymin=0, ymax=1, ls='--')
    plt.axvline(x=50, ymin=0, ymax=1, ls='--')
    plt.ylim(0.8*mean, 1.2*mean)
    plt.plot(x, means)
    plt.xlabel('Trial number')
    tit = 'Convergence for Bandit, mean=%d' % mean
    plt.title(tit)

    plt.ylabel('p_estimate')
    means = pd.DataFrame(means)
    # print('Mean estimate', means.values[-1])
    # print('Error of mean', mean - means.values[-1])
    # print('% Error      ', (mean - means.values[-1])/mean*100,'%')
    return means, die


# means, die = die_converge(1, 900)

#%%


def sim(mean, n_rolls):
    """Return how many predicted_means are still outside 5% band of mean.

    Parameters
    ----------
    mean: float,
        mean to converge to after n_rolls
    n_rolls: integer
        # turns for one die instance to converge

    Returns
    -------
    errors: dataframe
        how many predicted_mean of n_rolls are outside of a 5% margin of mean
        after the first number of rolls in trials trials_to_converge

        example:
            - the first column will be how many predicted_means lie outside 5%
            band after from trial 50 to last trial of die-converge
            - the second column will be howmany predicted_means lie outside 5%
            band after trial 100
    """
    trials_to_converge = [50, 100, 200, 300, 400, 500]
    means, _ = die_converge(mean, n_rolls)
    outliers = []

    for i in trials_to_converge:
        n_outside_range = sum(abs(means.iloc[i:, ].values - mean)/mean > 0.05)
        outliers.append(n_outside_range)
#    print(errors)
    errors = pd.DataFrame(data=[outliers], columns=trials_to_converge)
    return errors


def multiple_sims(mean, n_rolls, m):
    """Returns dataframe of errors after m runs of sim"""
    errors = sim(mean, n_rolls)
    for i in range(m-1):
        errors = pd.concat([errors, sim(mean, n_rolls)])
    errors.index = np.arange(m)
    print('---------- ---------- ---------- ----------')
    print('Mean:', mean, '    n_rolls:', n_rolls)
    print('# predicted_means off by >5% after trial #__')
    print(errors)
    return errors


# When mean=5, we need to go <200 steps for p_estimate to be inside 5%
#errors = multiple_sims(5, 1000, 15)

# However, when mean=2, we need to go a 300-400 steps for p_estimate to be
# closer but still not within the 5% range; at 1000 steps, almost all
# simulations are within 5% range
errors = multiple_sims(2, 1000, 15)

# mean of 1 is very noisy and needs lotsoftime toconverge, > 1000
# turnssimulations are within 5% range
# errors = multiple_sims(1, 1000, 15)

#%%


def experiment(u_1, u_2, u_3):

    dice = [Bandit(u_1, 'u1 = %s' % str(u_1)),
            Bandit(u_2, 'u2 = %s' % str(u_2)),
            Bandit(u_3, 'u3 = %s' % str(u_3))]

    sample_points = [5, 10, 20, 50, 100] #, 500, 1000, 1999]

    for i in range(NUM_TRIALS):
        highest_die = None
        maxsample = -20
        samples = []

        for d in dice:
            test = d.sample()
            samples.append(round(test, 3))
            if test > maxsample:
                maxsample = test
                highest_die = d

        highest_die.update(highest_die.roll(), debug=False)

        if i + 1 in sample_points:
            # print("current samples: %s" % allsamples)
            # print('Trial', i+1)
            plot(dice, i + 1, samples)

    return dice


def roll_count(u1, u2, u3, n_runs):
    """Return dataframe of n_runs rows.

    Parameters
    ----------
    u1 : float
        mean of Bandit1 in experiment().
    u2 : float
        mean of Bandit2 in experiment().
    u3 : float
        mean of Bandit3 in experiment().
    n_runs : integer
        number of runs of experiment().

    Returns
    -------
    rollcount : a dataframe of n_runs rows x 3 columns
        - column 1 is # rolls in trial of # row where die 1 was chosen i.e.,
        had highest sample
        - column two is # rolls in trial for that row where die 2 was chosen
        - column 3, where die 3 was chosen

    """
    rollcount = pd.DataFrame()

    for j in range(n_runs):
        dice = experiment(u1, u2, u3)
        rollcount = rollcount.append(np.transpose(pd.DataFrame([dice[i].rolls - 1 for i in range(3)])))

    # column three, ... etc.
    return rollcount


if __name__ == '__main__':

# OPTION 1 - for single run of experiment, uncomment below
    dice = experiment(1.5, 2.0, 2.3)
    print_dice(dice)


#%%
# OPTION 2 - RUN THIS CELL to see how difference of means affects
# optimal_chosen_pct
# Note - Algorithms for the multi-armed bandit problem 32 pg under references
# section above shows that # of bandits and variance are the only two criteria
# separating properly tuned bandit algorithms. Intuitively, greater difference
# in means between bandits for fixed variance will lead to faster convergence
# to the bandit with the higher real mean, since it will be less and less
# likely to generate a sample from the lower mean bandit that exceeds the
# sample of the higher mean bandit.  See some results below.

    n = 10
    diff = 1 * np.linspace(.1, 1, 10)
    percentages = []
    rewards = []

    for j in diff:
        u1 = 1.8
        u2 = 2.0
        u3 = 2.5
        rc = roll_count(u1, u2, u3 + j, n)
        rc.index = np.arange(n)
        # last die has highest mean, so we take mean of second column
        # of rollcount

        # optimal_chosen_pct is mean of # of times die 3 was chosen in the
        # n runs of roll_count()
        optimal_chosen_pct = rc[2].mean()
        reward = round(rc[0].mean() * u1 + rc[1].mean() * u2
                       + rc[2].mean() * (u3 + j), 2)
        # print('\nAfter %s runs, optimal Bandit chosen %.1f%% of time when mean='
        #  % (n, optimal_chosen_pct, u3 + j))
        percentages.append(optimal_chosen_pct)
        rewards.append(reward)

    percentages = np.transpose(pd.DataFrame(percentages))
    percentages = pd.concat([percentages, np.transpose(pd.DataFrame(rewards))])
    percentages.columns = diff + 2.5
    percentages.index = ['optimal_choice_pct', 'Expected reward']
    print('\n', n, 'Runs of 100 turn simulation')
    print('Highest mean bandit is column header  |  Second highest mean:', u2)
    print(percentages)
    print('Roll counts of each of ', n, 'trials')
    print(rc)
    percentages.to_html('temp.html')

"""
RESULTS
    Larger spread between means of bandits does increase
    slightly the % of correct optimal choices

 10 Runs of 100 turn simulation
Highest mean bandit is column header  |  Second highest mean: 2.0
                      2.6     2.7     2.8  ...     3.3    3.4     3.5
optimal_choice_pct   72.3   63.60   65.30  ...   69.70   84.4   69.60
Expected reward     241.7  242.64  247.58  ...  288.45  317.0  302.22

[2 rows x 10 columns]
Roll counts of each of  10 trials
    0   1    2
0  17   4   79
1   0   0  100
2  62  34    4
3   0   0  100
4  14  85    1
5   0   1   99
6   1   1   98
7  11  52   37
8   2   1   97
9   2  17   81

----------------------------------------

 10 Runs of 100 turn simulation
Highest mean bandit is column header  |  Second highest mean: 2.0
                       2.7     2.9     3.1  ...     4.1    4.3     4.5
optimal_choice_pct   75.90   78.10   70.60  ...   74.80   85.6   95.30
Expected reward     252.15  268.53  274.98  ...  356.52  395.2  437.55

[2 rows x 10 columns]
Roll counts of each of  10 trials for highest mean = 4.5
    0   1    2
0   1   1   98
1   0   1   99
2   1   0   99
3   0   0  100
4   0   0  100
5  33   0   67
6   0   0  100
7   0   0  100
8   0   0  100
9   0  10   90
"""
