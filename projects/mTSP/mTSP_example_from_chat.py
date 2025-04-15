# When agents in a Multi-Agent Traveling Salesman Problem (mTSP) start from different fixed locations
# (i.e., multiple depots), the problem becomes a variant known as the
# Multi-Depot Multiple Traveling Salesman Problem (MD-mTSP). In this scenario,
# each agent begins from its designated depot, and the collective goal is to visit
# all cities exactly once, optimizing for criteria such as minimizing the total distance
# traveled or balancing the workload among agents

# https://neos-guide.org/case-studies/tra/multiple-traveling-salesman-problem-mtsp/?utm_source=chatgpt.com
# The requirements on the set of routes are:
#
#     All of the routes must start and end at the (same) depot.
#     Each city must be visited exactly once by only one salesman.

# Multiple depots: Instead of one depot, the multi-depot mTSP has a set of depots, with mj salesmen at
# each depot j.
#   In the fixed destination version, a salesman returns to the same depot from which he started.
#   In the non-fixed destination version, a salesman does not need to return to the same depot
#   from which he started but the same number of salesmen must return as started from a particular depot.


# HERE non-fixed destination

# TO CHECK: Bektas, T. 2006. The multiple traveling salesman problem: an overview of
# formulations and solution procedures. OMEGA: The International Journal
# of Management Science 34(3), 209-219.

# check this
# https://chatgpt.com/share/67f8c56b-3c10-8010-bd0e-8c8177396312

# Cities: 10 cities labeled from 0 to 9.
# Agents: 2 agents starting from fixed depots (city 0 and city 1).
# Objective: Assign cities to agents such that each city is visited exactly once,
#   and the total distance traveled by all agents is minimized.
#   Agents do not need to return to their starting depots.

# %%
import random
import numpy as np
from deap import base, creator, tools, algorithms

# %% Define the number of cities and agents
NUM_CITIES = 10
NUM_AGENTS = 2
DEPOTS = [0, 1]  # Starting depots for each agent

# global prob variables

CX_PROB = 0.5
MUT_PROB = 0.2


# Generate random coordinates for cities
city_coords = np.random.rand(NUM_CITIES, 2)

# Compute the distance matrix between cities
distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)

# %%

# Create the fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# %%
# Attribute generator: random permutation of cities excluding depots
def create_individual():
    cities = list(set(range(NUM_CITIES)) - set(DEPOTS))
    random.shuffle(cities)
    # Split cities between agents
    split = len(cities) // NUM_AGENTS
    individual = []
    for i in range(NUM_AGENTS):
        route = (
            cities[i * split : (i + 1) * split]
            if i < NUM_AGENTS - 1
            else cities[i * split :]
        )
        individual.append(route)
    return individual


toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Evaluation function
def evaluate(individual):
    total_distance = 0
    for i, route in enumerate(individual):
        if route:
            # Start from depot
            distance = distance_matrix[DEPOTS[i]][route[0]]
            # Traverse the route
            for j in range(len(route) - 1):
                distance += distance_matrix[route[j]][route[j + 1]]
            total_distance += distance
    return (total_distance,)


toolbox.register("evaluate", evaluate)


# Crossover operator: swap routes between two individuals
# cxpb is taken from global variable


def crossover(ind1, ind2):
    for i in range(len(ind1)):
        if random.random() < CX_PROB:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# Mutation operator: shuffle cities within an agent's route
# mutpb is taken from global variable
def mutate(individual):
    for route in individual:
        if len(route) > 1 and random.random() < MUT_PROB:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
    return (individual,)


toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


# Main function
def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=100,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Print the best solution
    best = hof[0]
    print(f"Best total distance: {evaluate(best)[0]:.2f}")
    for i, route in enumerate(best):
        print(f"Agent {i+1} route: {DEPOTS[i]} -> {' -> '.join(map(str, route))}")


if __name__ == "__main__":
    main()
