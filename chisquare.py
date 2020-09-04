"""
A/B test Bayesian probability in python on Udemy.

Created on Wed Aug  5 19:30:12 2020
August 2020

@author: pjrowe2012


# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
# https://github.com/lazyprogrammer/machine_learning_examples/tree/master/ab_testing

# contingency table
#        click       no click
#------------------------------
# ad A |   a            b
# ad B |   c            d
#
# chi^2 = (ad - bc)^2 (a + b + c + d) / [ (a + b)(c + d)(a + c)(b + d)]
# degrees of freedom = (#cols - 1) x (#rows - 1) = (2 - 1)(2 - 1) = 1

# short example

# T = np.array([[36, 14], [30, 25]])
# c2 = np.linalg.det(T)**2 * T.sum() /
( T[0].sum()*T[1].sum()*T[:,0].sum()*T[:,1].sum() )
# p_value = 1 - chi2.cdf(x=c2, df=1)

# equivalent:
# (36-31.429)**2/31.429+(14-18.571)**2/18.571 + (30-34.571)**2/34.571
+ (25-20.429)**2/20.429

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import pandas as pd


class DataGenerator:

    def __init__(self, p1, p2):
        self.p1 = p1  # prob of click for ad 1
        self.p2 = p2  # prob of click for ad 2

    def next(self):
        click1 = 1 if (np.random.random() < self.p1) else 0
        click2 = 1 if (np.random.random() < self.p2) else 0
        return click1, click2


def get_click(p1, p2):
    """Generate a click for the simulation, replace DataGenerator."""
    click1 = 1 if (np.random.random() < p1) else 0
    click2 = 1 if (np.random.random() < p2) else 0
    return click1, click2


def get_p_value(T):
    # T is contingency table; np.array
    # same as scipy.stats.chi2_contingency(T, correction=False)
    # chi2_contingency(T,correction=False)
    # Out[71]:
    # (20.665367632599796, 5.469641027718189e-06, 1, array([[ 676., 1324.],
    #       [ 676., 1324.]]))
    T = np.matrix(T) if (type(T) == pd.DataFrame) else T

    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    # addressing arrays                sum of row1   row 2     sum col0   col1
    numerator = float(det) * det * T.sum()
    denominator = T[0].sum() * T[1].sum() * T[:, 0].sum() * T[:, 1].sum()
    c2 = numerator / denominator
    p = 1 - chi2.cdf(x=c2, df=1)
    return p


def run_experiment(prob1, prob2, N):
    """Run experiment.

    Parameters
    ----------
    prob1 : float
        click through rate of ad1.
    prob2 : float
        click through rate of ad2.
    N : integer
        number of trials.

    Returns
    -------
    None.

    """
    p_values = np.empty(N)
    T = np.zeros((2, 2)).astype(np.float32)
    for i in range(N):
        # c1, c2 = data.next()
        c1, c2 = get_click(prob1, prob2)
        # row 0 is for results of click trhus from ad1, col 1 = click,
        # col 0 = no click
        T[0, c1] += 1
        T[1, c2] += 1

        # need enough runs or samples so that sum of row or columns is nonzero
        if i < 10:
            p_values[i] = None
        else:
            p_values[i] = get_p_value(T)
    plt.plot(p_values)
    plt.plot(np.ones(N) * 0.05)
    plt.show()
    print('The contingency matrix is:')
    print('--------------------------')
    print(pd.DataFrame(T, index=['Ad A', 'Ad B'],
                       columns=['No click', 'Click']))


run_experiment(0.1, 0.11, 20000)

# simulation using clicks as shown in file

df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action'].values
b = b['action'].values

cont_table = pd.DataFrame([[a.sum(), len(a) - a.sum()],
                           [b.sum(), len(b) - b.sum()]])

get_p_value(cont_table)

# Out[134]: 0.0013069502732125926
# this is a significant p value as it is less than 0.05

"""
# SUGGESTED SOLUTION FROM COURSE CREATOR
#

import pandas as pd
x = pd.read_csv("advertisement_clicks.csv")

y=pd.pivot_table(x,index=["advertisement_id"])
y['click']=y['action']*len(x)
y['noclick']=2000-y['click']
T=np.array(y.iloc[:,1:])
get_p_value(T)
# x2 stat is 5.47e-6, which is obviously significant

## provided solution
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action']
b = b['action']

A_clk = a.sum()
A_noclk = a.size - a.sum()
B_clk = b.sum()
B_noclk = b.size - b.sum()

T = np.array([[A_clk, A_noclk], [B_clk, B_noclk]])

print(get_p_value(T))
"""
