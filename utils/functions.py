import numpy as np
import pandas as pd
from utils.constants import WEEKS, CAPITAL


def close_to_zero(x):
    return x <= 1e-8


def get_returns(stocks):
    returns = stocks.pct_change()
    return returns.fillna(np.mean(returns))


def random_population(n_assets, population_size):
    weights = np.random.uniform(0, 1, size=(population_size, n_assets))
    return weights / weights.sum(axis=-1).reshape((-1, 1))


def annualized_portfolio_return(returns, weights):
    weighted_returns = np.matmul(weights, np.mean(returns.values, 0))
    return (weighted_returns + 1) ** WEEKS - 1


def annualized_portfolio_volatility(returns, weights):
    variance = np.sum(weights * np.matmul(weights, np.cov(returns.T.values)), -1)
    return np.sqrt(variance) * np.sqrt(WEEKS)


def annualized_portfolio_performance(returns, weights):
    return np.stack([
        annualized_portfolio_return(returns, weights),
        annualized_portfolio_volatility(returns, weights)
    ], -1)


def print_ga_allocation(data, allocations, prices):
    print("\n\nPORTOFOLIUL OPTIM")
    for ticker_id in np.argsort(-allocations):
        print('%s - %.2f, %.2f lei, %d acțiuni' % (
            data.columns[ticker_id], allocations[ticker_id] * 100, CAPITAL * allocations[ticker_id],
            (CAPITAL * allocations[ticker_id]) / prices[data.columns[ticker_id]]))


def print_rs_allocation(portfolio_returns, portfolio_risks, sharpe_ratios, portfolio_weights, portfolio_tickers):
    portfolio_metrics = [portfolio_returns, portfolio_risks, sharpe_ratios, portfolio_weights, portfolio_tickers]
    portfolio_df = pd.DataFrame(portfolio_metrics).T
    portfolio_df.columns = ['Randament (%)', 'Risc (%)', 'Raport Sharpe (%)', 'Ponderi (%)', 'Emitenți']
    max_sharpe = portfolio_df.loc[portfolio_df['Raport Sharpe (%)'].astype(float).idxmax()]
    print('\nPORTOFOLIUL OPTIM')
    print(max_sharpe)


def rs_weights(sharpe_ratios, portfolio_weights):
    portfolio_metrics = [sharpe_ratios, portfolio_weights]
    portfolio_df = pd.DataFrame(portfolio_metrics).T
    max_sharpe = portfolio_df.loc[portfolio_df[0].astype(float).idxmax()]
    solution = max_sharpe[1]
    return solution


def print_time(stop, start):
    print('\nDurata: %.2f' % (stop - start))

import os
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')
