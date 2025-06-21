import itertools
import numpy as np


def compute_distance(route, distance_matrix):
    """Compute the total distance of a given route."""
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    return distance


def generate_balanced_partitions(cities, num_agents):
    """
    Generate all balanced partitions of 'cities' among 'num_agents'.
    Each agent receives either floor(n/k) or ceil(n/k) cities.
    """
    n = len(cities)
    base_size = n // num_agents
    remainder = n % num_agents

    # Create a list indicating the size of each partition
    # The first 'remainder' partitions will have one extra element
    partition_sizes = [
        base_size + 1 if i < remainder else base_size for i in range(num_agents)
    ]

    def helper(remaining_cities, sizes):
        if not sizes:
            yield []
            return
        first_size = sizes[0]
        for group in itertools.combinations(remaining_cities, first_size):
            remaining = list(remaining_cities)
            for city in group:
                remaining.remove(city)
            for rest in helper(remaining, sizes[1:]):
                yield [list(group)] + rest

    return helper(cities, partition_sizes)


def generate_partitions(cities, num_agents):
    """Generate all possible partitions of cities among agents."""

    def helper(cities, k):
        if k == 1:
            yield [cities]
        else:
            for i in range(1, len(cities)):
                for tail in helper(cities[i:], k - 1):
                    yield [cities[:i]] + tail

    return helper(cities, num_agents)


def grid_search_mTSP(distance_matrix, depots):
    num_agents = len(depots)
    num_cities = len(distance_matrix)
    cities = [i for i in range(num_cities) if i not in depots]
    min_total_distance = float("inf")
    best_routes = None

    for partition in generate_balanced_partitions(cities, num_agents):
        # Generate all permutations for each agent's assigned cities
        permutations = [list(itertools.permutations(p)) for p in partition]
        # Generate all combinations of these permutations
        for routes in itertools.product(*permutations):
            total_distance = 0
            for agent in range(num_agents):
                route = [depots[agent]] + list(routes[agent])
                total_distance += compute_distance(route, distance_matrix)
            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_routes = [[depots[i]] + list(routes[i]) for i in range(num_agents)]

    return best_routes, min_total_distance


import numpy as np

NUM_CITIES = 12
NUM_AGENTS = 4
DEPOTS = list(range(NUM_AGENTS))  # Starting depots for each agent
np.random.seed(123)
city_coords = np.random.rand(NUM_CITIES, 2)
distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)
print(distance_matrix)

best_routes, min_distance = grid_search_mTSP(distance_matrix, DEPOTS)
print("Best Routes:", best_routes)
print("Minimum Total Distance:", min_distance)
