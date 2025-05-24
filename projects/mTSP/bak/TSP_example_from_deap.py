#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

# https://github.com/DEAP/deap/blob/master/examples/ga/tsp.py


# %%
import array
import random
import json

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# %%

# gr*.json contains the distance map in list of list style in JSON format
# Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942
with open("gr24.json", "r") as tsp_data:
    tsp = json.load(tsp_data)

# %%

distance_map = tsp["DistanceMatrix"]
IND_SIZE = tsp["TourSize"]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# %%
# In the context of the TSP, where the goal is to find the shortest
# possible route that visits each city exactly once and returns to
# the origin city, the evalTSP function might be defined as follows:

# Here, distance_map is a precomputed matrix containing distances between cities.
# The function calculates the total distance of the tour represented by individual.
# The comma in return distance, ensures that the result is a tuple,
# as DEAP expects fitness values to be iterable.


def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return (distance,)


# In the DEAP library, tools.cxPartialyMatched implements
# the Partially Matched Crossover (PMX) operator, commonly
# used in genetic algorithms for permutation-based problems
# like the Traveling Salesman Problem (TSP). PMX ensures that
# offspring are valid permutations, preserving the uniqueness of elements.
toolbox.register("mate", tools.cxPartialyMatched)

# In the DEAP library, tools.mutShuffleIndexes is a mutation operator designed
# for permutation-based individuals, such as those used in problems like the
# Traveling Salesman Problem (TSP). This operator introduces variation by
# shuffling the positions of elements within an individual, helping to explore
# new permutations while maintaining the validity of the solution.
# indpb: The independent probability for each attribute to be exchanged with another position.
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

# In the DEAP (Distributed Evolutionary Algorithms in Python) library,
# tools.selTournament is a selection operator that implements tournament
# selection, a method commonly used in genetic algorithms to select individuals
# for reproduction based on their fitness.

# How tools.selTournament Works
# The selTournament function selects the best individuals from a population
# through a series of tournaments. Each tournament involves randomly choosing
# a subset of individuals and selecting the best among them. This process
# is repeated to select the desired number of individuals.

#     individuals: The list of individuals to select from.
#     k: The number of individuals to select.
#     tournsize: The number of individuals participating in each tournament.
#     fit_attr: The attribute of individuals to use as the selection criterion
#     (default is 'fitness').
#
# Selection Process:
#
#     For each of the k selections:
#
#         Randomly select tournsize individuals from the population.
#         deap.readthedocs.io+4GitHub+4GitHub+4
#
#         Evaluate their fitness using the specified fit_attr.
#
#         Select the individual with the best fitness among them.
#         GitHub

# This method ensures that individuals with higher fitness have a higher
# chance of being selected, while still maintaining diversity in the population.
toolbox.register("select", tools.selTournament, tournsize=3)

# In the DEAP (Distributed Evolutionary Algorithms in Python) framework,
# registering an evaluation function is a crucial step in setting up
# a genetic algorithm. The line toolbox.register("evaluate", evalTSP) binds
# the name "evaluate" to the function evalTSP within the toolbox.
toolbox.register("evaluate", evalTSP)

# %%


def main():
    random.seed(169)

    # In the DEAP (Distributed Evolutionary Algorithms in Python) library,
    # the line toolbox.population(n=300) is used to generate an initial population of
    # 300 individuals for the evolutionary algorithm.
    #
    # Role of the n Parameter
    # The n parameter in toolbox.population(n=300) specifies the number of individuals
    # to generate in the initial population. In this case, it creates a population of
    # 300 individuals. This approach allows for flexibility in defining the population size,
    # which can be adjusted based on the problem's complexity and computational resources.
    # By using toolbox.population(n=300), you efficiently generate
    # a diverse initial population, which is a crucial step in evolutionary algorithms
    # to ensure a broad search space and effective optimization.
    pop = toolbox.population(n=300)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # In the DEAP (Distributed Evolutionary Algorithms in Python) library,
    # algorithms.eaSimple is a built-in function that implements a basic
    # evolutionary algorithm. It's designed to facilitate rapid prototyping
    # by handling the main evolutionary loop, including selection, variation
    # (crossover and mutation), and evaluation processes.

    #     Key Parameters

    #     population: A list of individuals to evolve.
    #     toolbox: A DEAP toolbox object containing the registered evolutionary operators:
    #       - mate,
    #       - mutate,
    #       - select,
    #       - and evaluate.
    #     cxpb: The probability of mating two individuals (crossover probability).
    #     mutpb: The probability of mutating an individual (mutation probability).
    #     ngen: The number of generations to run the algorithm.
    #     stats (optional): A Statistics object to collect and report statistics during the evolution.
    #     halloffame (optional): A HallOfFame object to keep track of the best individuals found.
    #     verbose (optional): A boolean flag to control the display of statistics during evolution.
    # How It Works

    # The eaSimple function operates as follows:
    #     Initialization: Evaluate the initial population's fitness
    #     Evolution Loop: For each generation up to ngen
    #         Selection: Select individuals from the current population using the select operator
    #         Variation: Apply crossover (mate) and mutation (mutate) operators to produce offspring
    #         Evaluation: Evaluate the fitness of the new offspring
    #         Replacement: Replace the current population with the new offspring
    #         Statistics and Hall of Fame: Update statistics and the hall of fame if provided
    #     Return: After completing all generations, return the final population and a logbook containing the recorded statistics.

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, stats, hof


# %%
if __name__ == "__main__":
    res = main()
