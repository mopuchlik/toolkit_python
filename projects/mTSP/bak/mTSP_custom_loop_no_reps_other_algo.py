from deap import base, creator, tools, algorithms
import random
import numpy as np
import matplotlib.pyplot as plt


# Define the repair function
def repair(individual, num_cities, depots):
    """
    Repairs an individual to ensure each city is visited exactly once.
    """
    all_cities = set(range(num_cities)) - set(depots)
    visited = set()
    duplicates = []

    # Identify duplicates and collect visited cities
    for route in individual:
        for city in route:
            if city in visited:
                duplicates.append(city)
            else:
                visited.add(city)

    missing = list(all_cities - visited)
    random.shuffle(missing)

    # Replace duplicates with missing cities
    for route in individual:
        for i in range(len(route)):
            if route[i] in duplicates:
                if missing:
                    new_city = missing.pop()
                    duplicates.remove(route[i])
                    route[i] = new_city
    return individual


# Parameters
NUM_CITIES = 12
NUM_AGENTS = 4
DEPOTS = list(range(NUM_AGENTS))
N_GEN = 300
MU = 400  # Population size
LAMBDA = 400  # Number of children to produce at each generation
CX_PROB = 0.6
MUT_PROB = 0.3
N_POP = 400  # number of initial guesses/population

# Generate random coordinates for cities
np.random.seed(123)
city_coords = np.random.rand(NUM_CITIES, 2)
distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)

# Create the fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


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
def crossover(ind1, ind2):
    for i in range(len(ind1)):
        ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# Mutation operator: shuffle cities within an agent's route
def mutate(individual):
    for route in individual:
        if len(route) > 1:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
    return (individual,)


toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


def varOrWithRepair(population, toolbox, lambda_, cxpb, mutpb):
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover **or** mutation) and then repairing the offspring.
    """
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            toolbox.mate(ind1, ind2)
            ind = ind1
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            toolbox.mutate(ind)
        else:  # Reproduce without variation
            ind = toolbox.clone(random.choice(population))
        # Apply repair function
        repair(ind, NUM_CITIES, DEPOTS)
        offspring.append(ind)
    return offspring


def eaMuPlusLambdaWithRepair(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """
    This is the (μ + λ) evolutionary algorithm with a repair function.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)
    if halloffame is not None:
        halloffame.update(population)
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = varOrWithRepair(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# Main function with eaMuPlusLambda
def main():
    random.seed(42)
    pop = toolbox.population(n=N_POP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # Parameters for eaMuPlusLambda
    MU = N_POP
    LAMBDA = N_POP

    # Run the evolutionary algorithm
    pop, log = eaMuPlusLambdaWithRepair(
        pop,
        toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=N_GEN,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    # Print the best solution
    best = hof[0]
    print(f"Best total distance: {evaluate(best)[0]:.2f}")
    for i, route in enumerate(best):
        print(f"Agent {i+1} route: {DEPOTS[i]} -> {' -> '.join(map(str, route))}")


def plot_routes(city_coords, depots, individual, title="Routes"):
    plt.close("all")  # Close all existing figures
    plt.figure(figsize=(8, 6))
    # Plot all cities
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c="gray", label="Cities")
    # Highlight depots
    for i, depot in enumerate(depots):
        plt.scatter(
            city_coords[depot, 0], city_coords[depot, 1], c="red", label=f"Depot {i+1}"
        )
    # Assign distinct colors for each agent
    colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
    for i, route in enumerate(individual):
        if route:
            # Construct the full path starting from the depot
            path = [depots[i]] + route
            x = [city_coords[city][0] for city in path]
            y = [city_coords[city][1] for city in path]
            plt.plot(x, y, color=colors[i % len(colors)], label=f"Agent {i+1}")
    plt.title(title)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
