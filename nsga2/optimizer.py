import time
from nsga2.objectives import *
from nsga2.algorithm import non_dominated_fronts, tournament, crossover, mutation, selection
from utils.constants import RFR
from utils.functions import clearConsole

class Optimizer:

    def __init__(self, population_size=5000, max_iterations=100, mutation_p=0.2,
                 mutation_p_decay=0.98, mutation_sigma=0.01, verbose=False):
        self._population_size = population_size
        self._max_iter = max_iterations
        self._mutation_p = mutation_p
        self._mutation_p_decay = mutation_p_decay
        self._mutation_sigma = mutation_sigma
        self._verbose = verbose

    def run(self, returns, max_allocation=None):
        stats = {
            'return': {'min': [], 'max': [], 'avg': []},
            'volatility': {'min': [], 'max': [], 'avg': []},
            'constraints_violation': {'min': [], 'max': [], 'avg': []},
            'time_per_generation': []
        }
        returns_mean = np.mean(returns, 0)
        returns_cov = np.cov(returns.T)
        population = self._init_population(len(returns_mean))
        return_obj = annualized_return(population, returns_mean)
        volatility_obj = annualized_volatility(population, returns_cov)
        constraints_val = unit_sum_constraint(population)
        if max_allocation is not None:
            constraints_val += max_allocation_constraint(population, max_allocation)
        fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)

        for gen_idx in range(self._max_iter):
            gen_start_time = time.time()
            mutation_p = self._mutation_p * (self._mutation_p_decay ** gen_idx)

            # Generarea copiilor
            offspring = np.empty_like(population)
            for i in range(self._population_size):
                (p1_idx, p2_idx) = tournament(fronts, crowding_distances)
                offspring[i, :] = crossover(population[p1_idx], population[p2_idx])
                if np.random.uniform() < mutation_p:
                    offspring[i, :] = mutation(offspring[i, :], sigma=self._mutation_sigma)

            offspring = np.clip(offspring, 0, 1)

            # Populatia t+1
            population = np.concatenate((population, offspring), axis=0)
            return_obj = annualized_return(population, returns_mean)
            volatility_obj = annualized_volatility(population, returns_cov)
            constraints_val = unit_sum_constraint(population)
            if max_allocation is not None:
                constraints_val += max_allocation_constraint(population, max_allocation)
            fronts, crowding_distances = non_dominated_fronts(return_obj, volatility_obj, constraints_val)
            population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val = selection(
                population, fronts, crowding_distances, return_obj, volatility_obj, constraints_val)

            self._append_stats(stats, return_obj[fronts == 0], volatility_obj[fronts == 0],
                               constraints_val[fronts == 0], float(time.time() - gen_start_time))
            if self._verbose:
                print(gen_idx + 1, sep=' ', end=' ', flush=True)
#                print(
#                    'Generatia %d: Randament: %.2f | Risc: %.2f | Sharpe: %.2f' % (
#                        gen_idx + 1, stats['return']['avg'][-1] * 100,
#                        stats['volatility']['avg'][-1] * 100,
#                        ((stats['return']['max'][-1] - RFR) / stats['volatility']['max'][-1])
#                    ))

        pareto_front_ids = np.argwhere(fronts == 0).reshape((-1,))
        return population[pareto_front_ids], stats

    def _init_population(self, n_assets):
        population = np.random.uniform(0, 1, size=(self._population_size, n_assets))
        return population / np.sum(population, 1).reshape((-1, 1))

    def _append_stats(self, stats, return_obj, volatility_obj, constraints_val, tpg):
        stats['time_per_generation'].append(tpg)
        for (k, v) in [('return', return_obj), ('volatility', volatility_obj),
                       ('constraints_violation', constraints_val)]:
            stats[k]['min'].append(np.min(v))
            stats[k]['max'].append(np.max(v))
            stats[k]['avg'].append(np.mean(v))
