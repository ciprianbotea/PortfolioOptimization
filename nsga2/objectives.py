import numpy as np
from utils.constants import WEEKS


def annualized_return(solutions, returns_mean):
    returns = np.matmul(solutions, returns_mean)
    return (returns + 1) ** WEEKS - 1


def annualized_volatility(solutions, returns_cov):
    volatilities = np.sum(solutions * np.matmul(solutions, returns_cov), -1)
    return np.sqrt(volatilities * WEEKS)


def unit_sum_constraint(solutions, eps=1e-4):
    return np.clip(np.abs(np.sum(solutions, -1) - 1) - eps, 0, None)


def max_allocation_constraint(solutions, max_allocation, eps=1e-4):
    d = np.ones_like(solutions) * max_allocation
    return np.clip(np.sum(np.clip(solutions - d, 0, None), -1) - eps, 0, None)
