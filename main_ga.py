import timeit
import mysql.connector as msql
from nsga2.optimizer import Optimizer
from utils.functions import *
from utils.graphs import *
from utils.constants import *

# connection = msql.connect(host='localhost', database='bvb_stocks', user='root', password='')
# query = "Select * from closing_prices;"
# data = pd.read_sql(query, connection, index_col="date")
# query = "Select * from bet_index;"
# bet = pd.read_sql(query, connection, index_col="date")
# connection.close()

# Citire din fisier
data = pd.read_csv("data/bvb.csv", index_col=0)
bet = pd.read_csv("data/bet.csv", index_col=0)

stocks = data.select_dtypes(include=['float'])
market_index = bet.select_dtypes(include=['float'])

returns = get_returns(stocks)
index_return = get_returns(market_index)

random_weights = random_population(returns.shape[1], RANDOM_PORTFOLIOS)
random_solutions = annualized_portfolio_performance(returns, random_weights)

optimizer = Optimizer(mutation_sigma=1.0, verbose=True, max_iterations=MAX_ITER, population_size=POP_SIZE)

print('Algoritmul va rula timp de ' + str(MAX_ITER) + ' generatii.')

start = timeit.default_timer()
solutions, stats = optimizer.run(returns.values)
stop = timeit.default_timer()

optimal_value = annualized_portfolio_performance(returns, solutions)
highest_return = optimal_value[:, 0]
lowest_risk = optimal_value[:, 1]
sharpe_ratio = (highest_return - RFR) / lowest_risk
solution = solutions[np.argmax(sharpe_ratio)]
allocation = annualized_portfolio_performance(returns, solution)

# Vizualizare rezultate
print_ga_allocation(returns, solution, data.select_dtypes(include=['float']).iloc[-1])

for i in range(returns.shape[1]):
    portfolio_return = returns.select_dtypes(include=['float']).iloc[:, i] * solution[i]
    portfolio_return = portfolio_return * portfolio_return.shape[0] / WEEKS
plot_portfolio(portfolio_return, index_return)

plot_solutions(returns, solutions, random_solutions)

# cumulative_returns(returns)
returns_corelogram(returns)

print('RAPORTUL SHARPE: %.2f\n' % (np.amax(sharpe_ratio)))

print_time(stop, start)
show()
