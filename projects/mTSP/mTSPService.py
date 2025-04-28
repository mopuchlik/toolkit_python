# LINK TO CHAT
# https://chatgpt.com/share/680babd4-a68c-8010-b25e-4ea54bdadff5

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
import numpy.typing as npt
import pandas as pd
from typing import List, Tuple, Union
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


class mTSPService:
    def __init__(
        self,
        num_cities: int,  # number of places to arrive to
        num_agents: int,  # number of agents
        depots: list[int],  # Starting depots for each agent
        city_coords: npt.NDArray[np.float64],
        time_windows: List[Tuple[Union[float, np.float64], Union[float, np.float64]]],
        service_times: List[float],
        n_gen: int = 50,  # number of generations/iterations of the algorithm
        n_population: int = 1000,  # number of initial guesses/population
        cx_prob: float = 0.6,  # crossover probability
        mut_prob: float = 0.3,  # mutation probability
        theta: float = 1,  # parameter used to calculate time_to_distance = distance * theta
        penalty_const: float = 100,  # penalty constant for tresspassing time_window
    ):
        """
        Constructor
        """

        self.num_cities = num_cities
        self.num_agents = num_agents
        self.depots = depots
        self.city_coords = city_coords
        self.time_windows = time_windows
        self.service_times = service_times
        self.n_gen = n_gen
        self.n_population = n_population
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.theta = theta
        self.penalty_const = penalty_const

        # Compute and store the distance matrix once at the beginning
        self.distance_matrix = self.create_dist_matrix()

        # Initialize DEAP components
        self.initialize_deap()

    def create_dist_matrix(self) -> npt.NDArray[np.float64]:
        """
        Function calculates distance matrix based on self.city_coords
        """

        distance_matrix = np.linalg.norm(
            self.city_coords[:, np.newaxis] - self.city_coords, axis=2
        )

        return distance_matrix

    def time_to_distance(self, time):
        """
        Function calculates distance from time of travel (linear for now)
        """
        return time / self.theta

    def distance_to_time(self, distance):
        """
        Function calculates time of travel from distance (linear for now)
        """
        return distance * self.theta

    def initialize_deap(self):
        """
        Function initializing DEAP algorithm
        """

        # DEAP creator
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        # toolbox initialization
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.create_individual
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # stats initialization
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def create_individual(self):
        """
        Attribute generator: random permutation of cities excluding depots
        """
        cities = list(set(range(self.num_cities)) - set(self.depots))
        random.shuffle(cities)
        # Split cities between agents
        # Here it is determined that routes are balanced,
        split = len(cities) // self.num_agents
        individual = []
        # each agent gets the same share of cities + residual for the last agent
        for i in range(self.num_agents):
            route = (
                cities[i * split : (i + 1) * split]
                if i < self.num_agents - 1
                else cities[i * split :]
            )
            individual.append(route)
        return individual

    @staticmethod
    def crossover(ind1, ind2):
        """
        Crossover operator: swap routes between two individuals
        """
        for i in range(len(ind1)):
            ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    @staticmethod
    def mutate(individual):
        """
        Mutation operator: shuffle cities within an agent's route
        """
        for route in individual:
            if len(route) > 1:
                idx1, idx2 = random.sample(range(len(route)), 2)
                route[idx1], route[idx2] = route[idx2], route[idx1]
        return (individual,)

    def penalty(self, tresspass_time):
        """
        function calculating penalty for tresspasing time_window thresholds
        """
        penalty = tresspass_time * self.penalty_const

        return penalty

    def evaluate(self, individual):
        """
        Function evaluating metric that is being optimized by the algorithm.
        In this implementation eaSimple algrithm minimizes total distance.
        """

        total_distance = 0
        penalty = 0

        for i, route in enumerate(individual):
            if route:
                current_time = 0
                current_city = self.depots[i]
                for next_city in route:
                    travel_time = self.distance_to_time(
                        self.distance_matrix[current_city][next_city]
                    )
                    arrival_time = current_time + travel_time
                    start_window, end_window = self.time_windows[next_city]
                    if arrival_time > end_window:
                        # NOTE: algorithm minimizes distance so penalty is wrt to distance
                        penalty += self.time_to_distance(
                            self.penalty(arrival_time - end_window)
                        )  # Penalize late arrivals
                    elif arrival_time < start_window:
                        arrival_time = start_window  # Wait until the time window opens
                    current_time = arrival_time + self.service_times[next_city]
                    if current_time > end_window:
                        # NOTE: algorithm minimizes distance so penalty is wrt to distance
                        penalty += self.time_to_distance(
                            self.penalty(current_time - end_window)
                        )  # Penalize late finish
                    total_distance += self.distance_matrix[current_city][next_city]
                    current_city = next_city
        return (total_distance + penalty,)

    def repair(self, individual):
        """
        Repairs an individual to ensure each city is visited exactly once.
        """
        all_cities = set(range(self.num_cities)) - set(self.depots)
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

    def compute_agent_schedule(self, agent_idx, route):
        """
        Computes the schedule for a single agent.

        Parameters:
        - agent_idx: Index of the agent.
        - route: List of city indices assigned to the agent.
        - self.depots: List of depot indices for each agent.
        - self.distance_matrix: 2D numpy array with distances between cities.
        - self.theta: Scalar value to convert distance to time.
        - self.service_times: List of service times for each city.
        - self.time_windows: List of tuples (start_time, end_time) for each city.

        Returns:
        - A pandas DataFrame with the schedule for the agent.
        """
        depot = self.depots[agent_idx]
        schedule = []
        current_time = 0.0
        prev_location = depot

        # Start from depot
        schedule.append(
            {
                "Location": f"Depot {depot}",
                "Start in Depot": current_time,
                "Travel Time to City": 0.0,
                "Arrival Time in City": current_time,
                "Start Time in City": current_time,
                "Service Time in City": 0.0,
                "End Time in City": current_time,
                "Time Window": None,
            }
        )

        for city in route:
            travel_time = self.distance_to_time(
                self.distance_matrix[prev_location][city]
            )
            arrival_time = current_time + travel_time
            window_start, window_end = self.time_windows[city]

            # Check if arrival is before the time window starts
            if arrival_time < window_start:
                start_service_time = window_start
            else:
                start_service_time = arrival_time

            service_time = self.service_times[city]
            end_service_time = start_service_time + service_time

            # Annotate time window if binding
            time_window_annotation = None
            if arrival_time < window_start or arrival_time > window_end:
                time_window_annotation = (window_start, window_end)

            schedule.append(
                {
                    "Location": f"City {city}",
                    "Start in Depot": None,
                    "Travel Time to City": travel_time,
                    "Arrival Time in City": arrival_time,
                    "Start Time in City": start_service_time,
                    "Service Time in City": service_time,
                    "End Time in City": end_service_time,
                    "Time Window": time_window_annotation,
                }
            )

            current_time = end_service_time
            prev_location = city

        df = pd.DataFrame(schedule)
        return df

    def plot_routes(self, individual, title="Routes"):
        """
        Plotting function
        """

        plt.close("all")  # Close all existing figures
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        # Plot all cities
        plt.scatter(
            self.city_coords[:, 0], self.city_coords[:, 1], c="gray", label="Cities"
        )
        # Highlight depots
        for i, depot in enumerate(self.depots):
            plt.scatter(
                self.city_coords[depot, 0],
                self.city_coords[depot, 1],
                c="red",
                label=f"Depot {i+1}",
            )
        # Assign distinct colors for each agent
        colors = ["blue", "green", "orange", "purple", "cyan", "magenta"]
        for i, route in enumerate(individual):
            if route:
                # Construct the full path starting from the depot
                path = [self.depots[i]] + route
                x = [self.city_coords[city][0] for city in path]
                y = [self.city_coords[city][1] for city in path]
                plt.plot(x, y, color=colors[i % len(colors)], label=f"Agent {i+1}")

                # Calculate arrival and end times
                current_time = 0
                for j in range(1, len(path)):
                    prev_city = path[j - 1]
                    curr_city = path[j]
                    distance = np.linalg.norm(
                        self.city_coords[prev_city] - self.city_coords[curr_city]
                    )
                    travel_time = self.distance_to_time(distance)
                    arrival_time = current_time + travel_time
                    if arrival_time < self.time_windows[curr_city][0]:
                        arrival_time = self.time_windows[curr_city][0]
                    work_time = (
                        self.service_times[curr_city]
                        if self.service_times is not None
                        else 0
                    )
                    end_time = arrival_time + work_time

                    # Annotate city with arrival and end times
                    annotation_text = (
                        f"{curr_city}\nArr: {arrival_time:.1f}\nEnd: {end_time:.1f}"
                    )
                    text_color = "black"
                    # Check if time window is violated
                    if (
                        self.time_windows
                        and self.time_windows[curr_city][0] > 0
                        and self.time_windows[curr_city][1] < np.inf
                    ):
                        # if time_windows:
                        window_start, window_end = self.time_windows[curr_city]
                        if window_start <= arrival_time and end_time <= window_end:
                            text_color = "green"
                            annotation_text += (
                                f"\nWindow: [{window_start}, {window_end}]"
                            )
                        else:
                            text_color = "red"
                            annotation_text += (
                                f"\nWindow: [{window_start}, {window_end}]"
                            )

                    ax.annotate(
                        annotation_text,
                        (
                            self.city_coords[curr_city][0],
                            self.city_coords[curr_city][1],
                        ),
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

    def run_mTSP(self):
        random.seed(42)
        pop = self.toolbox.population(n=self.n_population)
        hof = tools.HallOfFame(1)

        # Evaluate the initial population
        for ind in pop:
            ind.fitness.values = self.toolbox.evaluate(ind)

        for gen in range(self.n_gen):
            # Select and clone the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Apply repair function
            for ind in offspring:
                self.repair(ind)

            # Evaluate the individuals with an invalid fitness
            # NOTE: check this
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Update the hall of fame with the generated individuals
            hof.update(offspring)
            if gen % 10 == 0:
                best = hof[0]
                plt.ion()  # Turn on interactive mode
                self.plot_routes(best, title=f"Generation {gen} Best Routes")
                plt.pause(1)  # Pause to update the plot
                plt.ioff()  # Turn off interactive mode after the loop

            # Replace the current population with the offspring
            pop[:] = offspring

            # Gather and print statistics
            record = self.stats.compile(pop)
            print(f"Gen {gen}: {record}")

        # Print the best solution
        best = hof[0]
        print(f"Best total distance: {self.evaluate(best)[0]:.2f}")
        for i, route in enumerate(best):
            print(
                f"Agent {i+1} route: {self.depots[i]} -> {' -> '.join(map(str, route))}"
            )

        agent_schedules = []
        for agent_idx, route in enumerate(best):
            df = self.compute_agent_schedule(agent_idx, route)
            agent_schedules.append(df)
            print(f"\nSchedule for Agent {agent_idx + 1}:\n")
            print(df.to_string(index=False))


# TEST RUN ############################

NUM_CITIES = 22
NUM_AGENTS = 5
DEPOTS = list(range(NUM_AGENTS))  # Starting depots for each agent
N_GEN = 51  # number of generations/iterations
N_POP = 10000  # number of initial guesses/population

# global prob variables
CX_PROB = 0.6  # crossover prob
MUT_PROB = 0.3  # mutation prob

np.random.seed(123)
CITY_COORDS = np.random.rand(NUM_CITIES, 2)

# # Compute the distance matrix between cities
# distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords, axis=2)
# # print(distance_matrix)

THETA = 1.0  # time = Distance * theta

# service times per city
SERVICE_TIMES = np.random.uniform(
    0.2, 1, size=NUM_CITIES
)  # Example: service times between 5 and 15 units

TIME_WINDOWS = [
    (0, float("inf")) for _ in range(NUM_CITIES)
]  # Default: no time window constraints
# Example: Set specific time windows for certain cities
TIME_WINDOWS[7] = (0.5, 1.5)
TIME_WINDOWS[9] = (0.75, 1.75)
TIME_WINDOWS[12] = (0.75, 1.75)
TIME_WINDOWS[13] = (0.5, 1.5)

if __name__ == "__main__":

    mstp_to_run = mTSPService(
        num_cities=NUM_CITIES,
        num_agents=NUM_AGENTS,
        depots=DEPOTS,
        city_coords=CITY_COORDS,
        time_windows=TIME_WINDOWS,
        service_times=SERVICE_TIMES,
        n_gen=N_GEN,
        n_population=N_POP,
        cx_prob=CX_PROB,
        mut_prob=MUT_PROB,
        theta=THETA,
        penalty_const=100,
    )

    mstp_to_run.run_mTSP()
