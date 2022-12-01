# This is a sample Python script.
from random import random
import numpy as np
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus
from scipy.optimize import linprog


# LP-based method
def lp_based_method():
    x11 = LpVariable("x11", 0, np.inf)
    x12 = LpVariable("x12", 0, np.inf)

    prob = LpProblem("myProblem", LpMinimize)
    prob += x11 + x12 <= 2

    prob += x11 + x12

    status = prob.solve()
    LpStatus[status]
    ...


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lp_based_method()
