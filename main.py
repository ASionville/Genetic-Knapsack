import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

## Running parameters
random.seed(75) # Set random seed for reproducibility
csv_file = 'objects.csv' # Path to the CSV file containing items
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file '{csv_file}' not found. Please provide a valid path.")
find_optimal = False # Set to True to find the optimal solution using exhaustive search
visualize = True # Set to True to visualize the evolution of population fitness

## Problem data
# Load items from objects.csv (CSV format: weight,value per line, no header)
items = []
with open('objects.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Skip header
    next(reader, None)
    for row in reader:
        value, weight = map(int, row)
        items.append((value, weight))

max_weight = 15
n_items = len(items)

## Genetic Algorithm parameters
population_size = 20
n_generations = 30
mutation_rate = 0.01
tournament_size = 3

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
print("Chromosome:", best)
print("Value:", evaluate(best))
print("Weight:", sum(best[i] * items[i][1] for i in range(n_items)))


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

    print("\nOptimal solution found:")
    print("Chromosome:", optimal_combination)
    print("Value:", optimal_value)
    print("Weight:", optimal_weight)
    print(f"Items included: {[i for i in range(n_items) if optimal_combination[i] == 1]}")

## Visualization of population fitness evolution
if visualize:
    print("\nVisualizing population fitness evolution...")

    plt.figure(figsize=(10, 6))
    plt.imshow(fitness_history, aspect='auto', cmap='viridis', vmin=0, vmax=77) # 77 is the max value for the example
    plt.colorbar(label='Fitness (Value)')
    plt.xlabel('Generation')
    plt.ylabel('Individual')
    plt.title('Population Fitness Evolution')
    plt.tight_layout()
    plt.show()