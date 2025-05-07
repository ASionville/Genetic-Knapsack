import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Genetic Algorithm for the Knapsack Problem")
parser.add_argument('--population_size', type=int, default=20, help='Size of the population (default: 20)')
parser.add_argument('--num_generations', type=int, default=30, help='Number of generations (default: 30)')
parser.add_argument('--mutation_rate', type=float, default=0.01, help='Mutation rate (default: 0.01)')
parser.add_argument('--tournament_size', type=int, default=3, help='Tournament size for selection (default: 3)')
parser.add_argument('--max_weight', type=int, default=15, help='Maximum weight of the knapsack (default: 15)')
parser.add_argument('--visualize', action='store_true', help='Visualize the population fitness evolution')
parser.add_argument('--exhaustive', action='store_true', help='Perform exhaustive search for the optimal solution')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--csv_file', type=str, default='objects.csv', help='Path to the CSV file containing items')
args = parser.parse_args()

# Running parameters
if args.seed is not None:
    random.seed(args.seed)
csv_file = args.csv_file
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please provide a valid path.")
find_optimal = args.exhaustive
visualize = args.visualize

# Problem data
items = []
with open(csv_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip header
    next(reader, None)
    for row in reader:
        value, weight = map(int, row)
        items.append((value, weight))

max_weight = args.max_weight
n_items = len(items)

# Genetic Algorithm parameters
population_size = args.population_size
n_generations = args.num_generations
mutation_rate = args.mutation_rate
tournament_size = args.tournament_size

## Genetic Algorithm functions

def generate_individual() -> list[int]:
    """
    Generate a random chromosome for the genetic algorithm.

    Returns:
        list[int]: A list of 0s and 1s representing the presence or absence of each item in the knapsack.
    """
    return [random.randint(0, 1) for _ in range(n_items)]

def evaluate(individual: list[int]) -> int:
    """
    Evaluate the fitness of a chromosome for the knapsack problem.

    Args:
        individual (list[int]): A chromosome (list of 0s and 1s).

    Returns:
        int: The total value of the items in the knapsack if the weight constraint is satisfied, otherwise 0.
    """
    value = sum(individual[i] * items[i][0] for i in range(n_items))
    weight = sum(individual[i] * items[i][1] for i in range(n_items))
    return value if weight <= max_weight else 0  # hard penalty

def tournament_selection(population: list[list[int]], k: int) -> list[int]:
    """
    Select an individual from the population using tournament selection.

    Args:
        population (list[list[int]]): The current population of chromosomes.
        k (int): The number of individuals to participate in the tournament.

    Returns:
        list[int]: The selected chromosome (winner of the tournament).
    """
    selected = random.sample(population, k)
    selected.sort(key=evaluate, reverse=True)
    return selected[0]

def crossover(parent1: list[int], parent2: list[int]) -> tuple[list[int], list[int]]:
    """
    Perform one-point crossover between two parent chromosomes.

    Args:
        parent1 (list[int]): The first parent chromosome.
        parent2 (list[int]): The second parent chromosome.

    Returns:
        tuple[list[int], list[int]]: Two offspring chromosomes resulting from the crossover.
    """
    point = random.randint(1, n_items - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(individual: list[int]) -> list[int]:
    """
    Apply bit-flip mutation to a chromosome.

    Args:
        individual (list[int]): The chromosome to mutate.

    Returns:
        list[int]: The mutated chromosome.
    """
    # x ^ 1 flips x (0 -> 1 or 1 -> 0)
    return [bit ^ 1 if random.random() < mutation_rate else bit for bit in individual]

## Main loop

# Initialize population
population = [generate_individual() for _ in range(population_size)]

# For visualization: store fitness values for each individual at each generation
fitness_history = np.zeros((population_size, n_generations), dtype=int)

for generation in range(n_generations):
    # Evaluation
    population = sorted(population, key=evaluate, reverse=True)
    # Store fitness for visualization
    for idx, individual in enumerate(population):
        fitness_history[idx, generation] = evaluate(individual)
    best = population[0]
    print(f"Generation {generation}: Best score = {evaluate(best)}")

    # New population
    new_population = [best]  # Elitism
    while len(new_population) < population_size:
        parent1 = tournament_selection(population, tournament_size)
        parent2 = tournament_selection(population, tournament_size)
        offspring1, offspring2 = crossover(parent1, parent2)
        new_population.append(mutate(offspring1))
        if len(new_population) < population_size:
            new_population.append(mutate(offspring2))
    
    population = new_population

# Final result
print("\nBest solution found:")
print("\tChromosome:", best)
print("\tValue:", evaluate(best))
print("\tWeight:", sum(best[i] * items[i][1] for i in range(n_items)))


## Exhaustive search to find the optimal solution
# This is only feasible for small n_items due to exponential complexity
# Generate all possible combinations of items (2^n_items)
# and evaluate their fitness
if find_optimal:
    print("\nFinding optimal solution using exhaustive search...")

    # Test all possible combinations of items
    # to find the optimal solution
    optimal_value = 0
    optimal_weight = 0
    optimal_combination = None
    for i in range(1 << n_items):
        combination = [(i >> j) & 1 for j in range(n_items)]
        value = evaluate(combination)
        weight = sum(combination[j] * items[j][1] for j in range(n_items))
        if value > optimal_value and weight <= max_weight:
            optimal_value = value
            optimal_weight = weight
            optimal_combination = combination

    print("\nOptimal solution found (exhaustive search):")
    print("\tChromosome:", optimal_combination)
    print("\tValue:", optimal_value)
    print("\tWeight:", optimal_weight)

## Visualization of population fitness evolution
if visualize:
    print("\nVisualizing population fitness evolution...")
    plt.figure(figsize=(10, 6))
    plt.imshow(fitness_history, aspect='auto', cmap='viridis', vmin=0, vmax=77)
    plt.colorbar(label='Fitness (Value)')
    plt.xlabel('Generation')
    plt.ylabel('Individual')
    plt.title('Population Fitness Evolution')
    plt.tight_layout()
    plt.show()