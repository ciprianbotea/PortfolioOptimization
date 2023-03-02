import pandas as pd
import numpy as np
import mysql.connector as msql
import timeit

from utils.functions import get_returns, print_time, print_rs_allocation, rs_weights
from utils.graphs import random_search, plot_portfolio, plot_solutions, show
from utils.constants import *

# connection = msql.connect(host='localhost', database='bvb_stocks', user='root', password='')
# query = "Select * from closing_prices;"
# data = pd.read_sql(query, connection, index_col="date")
# query = "Select * from bet_index;"
# bet = pd.read_sql(query, connection, index_col="date")
# connection.close()

# Citire din fisier
data = pd.read_csv("data/bvb.csv", index_col=0)
bet = pd.read_csv("data/bet.csv",  index_col=0)

stocks = data.select_dtypes(include=['float'])
market_index = bet.select_dtypes(include=['float'])
tickers = stocks.columns

returns = get_returns(stocks)
index_return = get_returns(market_index)

portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []
portfolio_weights = []
portfolio_tickers = []

start = timeit.default_timer()

for portfolio in range(RANDOM_PORTFOLIOS):
    # weights = np.full(stocks.shape[1], 1 / stocks.shape[1])
    weights = np.random.random_sample(stocks.shape[1])
    weights = np.round(weights / np.sum(weights), 2)
    portfolio_weights.append(weights)

    annualized_return = np.sum(returns.mean() * weights) * WEEKS
    portfolio_returns.append(round(annualized_return * 100, 2))

    covariance = returns.cov() * WEEKS
    variance = np.dot(weights.T, np.dot(covariance, weights))
    standard_deviation = np.sqrt(variance)
    portfolio_risks.append(round(standard_deviation * 100, 2))

    sharpe_ratio = round(((annualized_return - RFR) / standard_deviation), 2)
    sharpe_ratios.append(sharpe_ratio)

    portfolio_tickers.append(tickers)

stop = timeit.default_timer()

portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)
portfolio_weights = np.array(portfolio_weights)
portfolio_tickers = np.array(portfolio_tickers)

solution = rs_weights(sharpe_ratios, portfolio_weights)

for i in range(returns.shape[1]):
    portfolio_return = returns.select_dtypes(include=['float']).iloc[:, i] * solution[i]
    portfolio_return = portfolio_return * portfolio_return.shape[0] / WEEKS

plot_portfolio(portfolio_return, index_return)

print_rs_allocation(portfolio_returns, portfolio_risks, sharpe_ratios, portfolio_weights, portfolio_tickers)

print_time(stop, start)

random_search(portfolio_risks, portfolio_returns)

show()
