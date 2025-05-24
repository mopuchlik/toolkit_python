# %%
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Define the number of cities and salesmen
NUM_CITIES = 10
NUM_SALESMEN = 2
DEPOT = 0  # Index of the depot city

# %%

# Generate random coordinates for the cities
random.seed(42)
city_coords = [
    (random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)
]

# %%


# Compute the distance matrix
def compute_distance_matrix(coords):
    size = len(coords)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                matrix[i][j] = np.hypot(dx, dy)
    return matrix


distance_matrix = compute_distance_matrix(city_coords)

# %% Create the DEAP toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Individual representation: a permutation of city indices (excluding the depot)
toolbox.register("indices", random.sample, range(1, NUM_CITIES), NUM_CITIES - 1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# %%
# Evaluation function
def evaluate(individual):
    # Split the tour among salesmen
    split_size = len(individual) // NUM_SALESMEN
    tours = [
        individual[i * split_size : (i + 1) * split_size] for i in range(NUM_SALESMEN)
    ]
    # If there are remaining cities, add them to the last salesman
    remaining = len(individual) % NUM_SALESMEN
    if remaining:
        tours[-1].extend(individual[-remaining:])

    total_distance = 0
    for tour in tours:
        if tour:
            # Start from the depot
            prev_city = DEPOT
            for city in tour:
                total_distance += distance_matrix[prev_city][city]
                prev_city = city
            # Return to the depot
            total_distance += distance_matrix[prev_city][DEPOT]
    return (total_distance,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# %%


def main():
    population = toolbox.population(n=100)
    NGEN = 100
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual:", best_ind)
    print("Best fitness:", best_ind.fitness.values[0])


# %%

if __name__ == "__main__":
    main()
