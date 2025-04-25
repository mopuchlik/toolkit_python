# LINK TO CHAT
# https://chatgpt.com/share/67fe8471-84c4-8010-9f3d-6883cde974bd

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


# Cities: 10 cities labeled from 0 to 9.
# Agents: 2 agents starting from fixed depots (city 0 and city 1).
# Objective: Assign cities to agents such that each city is visited exactly once,
#   and the total distance traveled by all agents is minimized.
#   Agents do not need to return to their starting depots.

# %%
import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# %% Define the number of cities and agents
NUM_CITIES = 22
NUM_AGENTS = 5
DEPOTS = list(range(NUM_AGENTS))  # Starting depots for each agent
N_GEN = 51  # number of generations/iterations
N_POP = 10000  # number of initial guesses/population

# global prob variables
CX_PROB = 0.6  # crossover prob
MUT_PROB = 0.3  # mutation prob

np.random.seed(123)
city_coords = np.random.rand(NUM_CITIES, 2)

# Compute the distance matrix between cities
distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)
# print(distance_matrix)

theta = 1.0  # time = Distance * theta

# service times per city
service_times = np.random.uniform(
    0.2, 1, size=NUM_CITIES
)  # Example: service times between 5 and 15 units


time_windows = [
    (0, float("inf")) for _ in range(NUM_CITIES)
]  # Default: no time window constraints
# Example: Set specific time windows for certain cities
time_windows[7] = (0.5, 1.5)
time_windows[9] = (0.75, 1.75)
time_windows[12] = (0.75, 1.75)
time_windows[13] = (0.5, 1.5)

# %%
# def plot_routes(city_coords, depots, individual, title="Routes"):
#     plt.close("all")  # Close all existing figures
#     plt.figure(figsize=(8, 6))
#     # Plot all cities
#     plt.scatter(city_coords[:, 0], city_coords[:, 1], c="gray", label="Cities")
#     # Highlight depots
#     for i, depot in enumerate(depots):
#         plt.scatter(
#             city_coords[depot, 0], city_coords[depot, 1], c="red", label=f"Depot {i+1}"
#         )
#     # Assign distinct colors for each agent
#     colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
#     for i, route in enumerate(individual):
#         if route:
#             # Construct the full path starting from the depot
#             path = [depots[i]] + route
#             x = [city_coords[city][0] for city in path]
#             y = [city_coords[city][1] for city in path]
#             plt.plot(x, y, color=colors[i % len(colors)], label=f"Agent {i+1}")
#     plt.title(title)
#     plt.legend()
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.grid(True)
#     plt.show()


# def plot_routes(
#     city_coords,
#     depots,
#     individual,
#     title="Routes",
#     theta=1.0,
#     work_times=None,
#     time_windows=None,
# ):
#     if work_times is None:
#         work_times = np.zeros(NUM_CITIES)
#     if time_windows is None:
#         time_windows = {}

#     plt.close("all")
#     plt.figure(figsize=(10, 8))
#     plt.scatter(city_coords[:, 0], city_coords[:, 1], c="gray", label="Cities")

#     # Highlight depots
#     for i, depot in enumerate(depots):
#         plt.scatter(
#             city_coords[depot, 0], city_coords[depot, 1], c="red", label=f"Depot {i+1}"
#         )

#     colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
#     for i, route in enumerate(individual):
#         if route:
#             path = [depots[i]] + route
#             x = [city_coords[city][0] for city in path]
#             y = [city_coords[city][1] for city in path]
#             plt.plot(x, y, color=colors[i % len(colors)], label=f"Agent {i+1}")

#             current_time = 0.0
#             for j in range(1, len(path)):
#                 prev_city = path[j - 1]
#                 curr_city = path[j]
#                 distance = np.linalg.norm(
#                     city_coords[curr_city] - city_coords[prev_city]
#                 )
#                 travel_time = theta * distance
#                 arrival_time = current_time + travel_time

#                 # Check for time window
#                 if curr_city in time_windows:
#                     earliest, latest = time_windows[curr_city]
#                     if arrival_time < earliest:
#                         arrival_time = earliest  # Wait until the time window opens
#                     elif arrival_time > latest:
#                         print(
#                             f"Agent {i+1} arrives at city {curr_city} outside the time window."
#                         )

#                 end_time = arrival_time + work_times[curr_city]
#                 current_time = end_time

#                 # Annotate the city with arrival and end times
#                 plt.annotate(
#                     f"{arrival_time:.1f}/{end_time:.1f}",
#                     (city_coords[curr_city][0], city_coords[curr_city][1]),
#                     textcoords="offset points",
#                     xytext=(0, 10),
#                     ha="center",
#                     fontsize=8,
#                 )

#     plt.title(title)
#     plt.legend()
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.grid(True)
#     plt.show()


# def plot_routes(
#     city_coords, depots, individual, title="Routes", theta=1.0, work_times=None
# ):
#     plt.close("all")  # Close all existing figures
#     plt.figure(figsize=(8, 6))
#     ax = plt.gca()
#     # Plot all cities
#     plt.scatter(city_coords[:, 0], city_coords[:, 1], c="gray", label="Cities")
#     # Highlight depots
#     for i, depot in enumerate(depots):
#         plt.scatter(
#             city_coords[depot, 0], city_coords[depot, 1], c="red", label=f"Depot {i+1}"
#         )
#     # Assign distinct colors for each agent
#     colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
#     for i, route in enumerate(individual):
#         if route:
#             # Construct the full path starting from the depot
#             path = [depots[i]] + route
#             x = [city_coords[city][0] for city in path]
#             y = [city_coords[city][1] for city in path]
#             plt.plot(x, y, color=colors[i % len(colors)], label=f"Agent {i+1}")
#             # Annotate each city with arrival and end times
#             current_time = 0.0
#             for j in range(1, len(path)):
#                 prev_city = path[j - 1]
#                 curr_city = path[j]
#                 distance = np.linalg.norm(
#                     city_coords[prev_city] - city_coords[curr_city]
#                 )
#                 travel_time = theta * distance
#                 arrival_time = current_time + travel_time
#                 work_time = work_times[curr_city] if work_times is not None else 0.0
#                 end_time = arrival_time + work_time
#                 current_time = end_time
#                 # Annotate the current city
#                 ax.annotate(
#                     f"{curr_city}\nA:{arrival_time:.2f}\nE:{end_time:.2f}",
#                     (city_coords[curr_city][0], city_coords[curr_city][1]),
#                     textcoords="offset points",
#                     xytext=(0, 10),
#                     ha="center",
#                     fontsize=8,
#                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
#                 )
#     plt.title(title)
#     plt.legend()
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.grid(True)
#     plt.show()
def plot_routes(
    city_coords,
    depots,
    individual,
    title="Routes",
    theta=1.0,
    work_times=None,
    time_windows=None,
):
    plt.close("all")  # Close all existing figures
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
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

            # Calculate arrival and end times
            current_time = 0
            for j in range(1, len(path)):
                prev_city = path[j - 1]
                curr_city = path[j]
                distance = np.linalg.norm(
                    city_coords[prev_city] - city_coords[curr_city]
                )
                travel_time = theta * distance
                arrival_time = current_time + travel_time
                if arrival_time < time_windows[curr_city][0]:
                    arrival_time = time_windows[curr_city][0]
                work_time = work_times[curr_city] if work_times is not None else 0
                end_time = arrival_time + work_time

                # Annotate city with arrival and end times
                annotation_text = (
                    f"{curr_city}\nArr: {arrival_time:.1f}\nEnd: {end_time:.1f}"
                )
                text_color = "black"
                # Check if time window is violated
                if (
                    time_windows
                    and time_windows[curr_city][0] > 0
                    and time_windows[curr_city][1] < np.inf
                ):
                    # if time_windows:
                    window_start, window_end = time_windows[curr_city]
                    if window_start <= arrival_time and end_time <= window_end:
                        text_color = "green"
                        annotation_text += f"\nWindow: [{window_start}, {window_end}]"
                    else:
                        text_color = "red"
                        annotation_text += f"\nWindow: [{window_start}, {window_end}]"

                ax.annotate(
                    annotation_text,
                    (city_coords[curr_city][0], city_coords[curr_city][1]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    color=text_color,
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                )

                current_time = end_time
    plt.title(title)
    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


# %% Generate random coordinates for cities

plt.ion()  # Turn on interactive mode
plot_routes(
    city_coords,
    DEPOTS,
    [[] for _ in range(NUM_AGENTS)],
    title="Initial City Layout",
    theta=1,
    work_times=service_times,
    time_windows=time_windows,
)
plt.pause(1)  # Pause to update the plot
plt.ioff()  # Turn off interactive mode after the loop


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
# def evaluate(individual):
#     total_distance = 0
#     for i, route in enumerate(individual):
#         if route:
#             # Start from depot
#             distance = distance_matrix[DEPOTS[i]][route[0]]
#             # Traverse the route
#             for j in range(len(route) - 1):
#                 distance += distance_matrix[route[j]][route[j + 1]]
#             total_distance += distance
#     return (total_distance,)


def evaluate(individual):
    total_distance = 0
    penalty = 0
    for i, route in enumerate(individual):
        if route:
            current_time = 0
            current_city = DEPOTS[i]
            for next_city in route:
                travel_time = theta * distance_matrix[current_city][next_city]
                arrival_time = current_time + travel_time
                start_window, end_window = time_windows[next_city]
                if arrival_time > end_window:
                    penalty += (
                        (arrival_time - end_window) / theta
                    ) * 100  # Penalize late arrivals, may be other function
                elif arrival_time < start_window:
                    arrival_time = start_window  # Wait until the time window opens
                current_time = arrival_time + service_times[next_city]
                if current_time > end_window:
                    penalty += (
                        (current_time - end_window) / theta
                    ) * 100  # Penalize late finish, may be other function
                total_distance += distance_matrix[current_city][next_city]
                current_city = next_city
    return (total_distance + penalty,)


toolbox.register("evaluate", evaluate)


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


# Main function with custom evolutionary loop
def main():
    random.seed(42)
    pop = toolbox.population(n=N_POP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Evaluate the initial population
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(N_GEN):
        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Apply repair function
        for ind in offspring:
            repair(ind, NUM_CITIES, DEPOTS)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        if gen % 10 == 0:
            best = hof[0]
            # plt.ion()  # Turn on interactive mode
            plot_routes(
                city_coords,
                DEPOTS,
                best,
                title=f"Generation {gen} Best Routes",
                theta=1,
                work_times=service_times,
                time_windows=time_windows,
            )
            # plt.pause(1)  # Pause to update the plot
            # plt.ioff()  # Turn off interactive mode after the loop

        # Replace the current population with the offspring
        pop[:] = offspring

        # Gather and print statistics
        record = stats.compile(pop)
        print(f"Gen {gen}: {record}")

    # Print the best solution
    best = hof[0]
    print(f"Best total distance: {evaluate(best)[0]:.2f}")
    for i, route in enumerate(best):
        print(f"Agent {i+1} route: {DEPOTS[i]} -> {' -> '.join(map(str, route))}")


if __name__ == "__main__":
    main()
