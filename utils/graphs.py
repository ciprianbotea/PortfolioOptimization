import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.functions import annualized_portfolio_performance
from utils.constants import WEEKS


def cumulative_returns(returns):
    plt.figure(figsize=(20, 12))
    plt.title('Randamentul cumulat al acțiunilor(%)', fontsize=20, font='Times New Roman')
    for c in returns.columns:
        plt.plot(np.cumprod(returns[c] + 1) - 1, label=c)
    plt.xlabel('Data')
    plt.ylabel('Randament cumulat')
    plt.legend(loc='upper left', fontsize=10)


def plot_portfolio(portfolio, index):
    plt.figure(figsize=(20, 12))
    plt.title('Randamentul portofoliului selectat de algoritmul genetic versus randamentul indicelui BET', fontsize=20, font='Times New Roman')
    plt.plot((np.cumprod(portfolio + 1) - 1), label='Portofoliu')
    plt.plot((np.cumprod(index + 1) - 1), label='Indicele BET')
    plt.xticks(np.arange(0, portfolio.shape[0], WEEKS/2), rotation=45)
    plt.xlabel('Data')
    plt.ylabel('Randament cumulat')
    plt.legend(loc='upper left', fontsize=10)


def plot_solutions(data, solutions, random_solutions):
    optimal_values = annualized_portfolio_performance(data, solutions)
    plt.figure(figsize=(20, 10))
    plt.title('Spațiul soluțiilor (aproximare cu un milion de elemente) și rezultatele calculate', fontsize=15, font='Times New Roman')
    plt.scatter(random_solutions[:, 1], random_solutions[:, 0], alpha=.5)
    plt.scatter(optimal_values[:, 1], optimal_values[:, 0], alpha=.5)
    plt.xlabel('Risc')
    plt.ylabel('Randament')


def returns_corelogram(returns):
    plt.figure(figsize=(15, 10))
    plt.title('Matricea de covarianță - nivelul dependendenței dintre active', fontsize=20, font='Times New Roman')
    cm = returns.corr()
    mask = (1 - np.tril(np.ones_like(cm))) == 1
    cm[np.eye(cm.shape[0]) == 1] = np.nan
    cm[mask] = np.nan
    sns.heatmap(cm, cmap="RdBu_r")


def random_search(portfolio_risks, portfolio_returns):
    plt.figure(figsize=(20, 10))
    plt.scatter(portfolio_risks, portfolio_returns,
                c=portfolio_returns / portfolio_risks, alpha=.5)
    plt.title('Portofoliile generate aleatoar', fontsize=20, font='Times New Roman')
    plt.xlabel('Risc', fontsize=10,)
    plt.ylabel('Randament', fontsize=10)
    plt.colorbar(label='Raport Sharpe')


def show():
    plt.show()
