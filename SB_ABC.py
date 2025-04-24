"""
Author: Kanchan Rajwar
This code is dedicated to find Structural bias of ABC algorithms and GA, PSO and DE algorithms through Generalised Signature Test.
please choose particulars pop_size (line 15), dimensions (line 16), and grid_cells (line 42).
This parameters are sensitive for Generalised Signature Test
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Common Parameters
pop_size = 100  # Population size
dimensions =2  # Search space dimensions
max_generations = 100  # Number of iterations
bounds = (0, 1)  # Search space bounds

# PSO Parameters
w = 0.5  # Inertia weight
c1 = 2  # Cognitive weight
c2 = 2  # Social weight
v_max = 1.0  # Maximum velocity

# ABC Parameters
limit = 20  # Trial limit for scout phase

# DE Parameters
F = 0.5  # Differential weight
CR = 0.9  # Crossover probability

# GA Parameters
p_c = 0.8  # Crossover probability
p_m = 0.1  # Mutation probability
tournament_size = 3  # Tournament selection size

def objective_function(solution):
    """Random objective function (minimize)."""
    return np.random.uniform(0, 1)

def calculate_ssf(population, dimensions, grid_cells=10):
    """Calculate SSF based on population density in a grid."""
    grid_size = grid_cells
    grid = np.zeros((grid_size,) * dimensions)
    for individual in population:
        indices = tuple((individual * grid_size).astype(int) % grid_size)
        grid[indices] += 1
    empty_hypercubes = np.sum(grid == 0)
    return float(empty_hypercubes / grid_size ** dimensions)

def initialize_population(pop_size, dimensions, bounds):
    """Initialize population within bounds."""
    return np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))

# PSO Implementation
def pso(pop_size, dimensions, max_generations, bounds):
    positions, velocities = initialize_population(pop_size, dimensions, bounds), np.random.uniform(-v_max, v_max, (pop_size, dimensions))
    personal_best_positions = positions.copy()
    personal_best_scores = np.array([objective_function(ind) for ind in positions])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = np.min(personal_best_scores)
    ssf_values = []

    for generation in range(max_generations):
        scores = np.array([objective_function(ind) for ind in positions])
        improved = scores < personal_best_scores
        personal_best_positions[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]
        if np.min(scores) < global_best_score:
            global_best_position = positions[np.argmin(scores)]
            global_best_score = np.min(scores)
        ssf_values.append(calculate_ssf(positions, dimensions))
        for i in range(pop_size):
            r1, r2 = np.random.rand(dimensions), np.random.rand(dimensions)
            cognitive = c1 * r1 * (personal_best_positions[i] - positions[i])
            social = c2 * r2 * (global_best_position - positions[i])
            velocities[i] = w * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])
    return ssf_values

# ABC Implementation
def abc(pop_size, dimensions, max_generations, bounds):
    population = initialize_population(pop_size, dimensions, bounds)
    values = np.array([objective_function(ind) for ind in population])
    fitness = np.array([1 / (1 + v) for v in values])  # Fitness for minimization
    trials = np.zeros(pop_size)
    global_best_idx = np.argmin(values)
    global_best_position = population[global_best_idx].copy()
    global_best_value = values[global_best_idx]
    ssf_values = []

    for generation in range(max_generations):
        # Employed Bee Phase
        for i in range(pop_size):
            j = np.random.randint(dimensions)
            k = np.random.choice([x for x in range(pop_size) if x != i])
            phi = np.random.uniform(-1, 1)
            candidate = population[i].copy()
            candidate[j] = population[i][j] + phi * (population[i][j] - population[k][j])
            candidate = np.clip(candidate, bounds[0], bounds[1])
            candidate_value = objective_function(candidate)
            candidate_fitness = 1 / (1 + candidate_value)
            if candidate_fitness > fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
                values[i] = candidate_value
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bee Phase
        probabilities = fitness / np.sum(fitness)
        for _ in range(pop_size):
            i = np.random.choice(range(pop_size), p=probabilities)
            j = np.random.randint(dimensions)
            k = np.random.choice([x for x in range(pop_size) if x != i])
            phi = np.random.uniform(-1, 1)
            candidate = population[i].copy()
            candidate[j] = population[i][j] + phi * (population[i][j] - population[k][j])
            candidate = np.clip(candidate, bounds[0], bounds[1])
            candidate_value = objective_function(candidate)
            candidate_fitness = 1 / (1 + candidate_value)
            if candidate_fitness > fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
                values[i] = candidate_value
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bee Phase
        max_trials_idx = np.argmax(trials)
        if trials[max_trials_idx] >= limit:
            population[max_trials_idx] = np.random.uniform(bounds[0], bounds[1], dimensions)
            values[max_trials_idx] = objective_function(population[max_trials_idx])
            fitness[max_trials_idx] = 1 / (1 + values[max_trials_idx])
            trials[max_trials_idx] = 0

        # Update global best
        best_idx = np.argmin(values)
        if values[best_idx] < global_best_value:
            global_best_position = population[best_idx].copy()
            global_best_value = values[best_idx]
        ssf_values.append(calculate_ssf(population, dimensions))
    return ssf_values

# DE Implementation
def de(pop_size, dimensions, max_generations, bounds):
    population = initialize_population(pop_size, dimensions, bounds)
    values = np.array([objective_function(ind) for ind in population])
    global_best_idx = np.argmin(values)
    global_best_position = population[global_best_idx].copy()
    global_best_value = values[global_best_idx]
    ssf_values = []

    for generation in range(max_generations):
        for i in range(pop_size):
            # Mutation
            indices = np.random.choice([x for x in range(pop_size) if x != i], 3, replace=False)
            r1, r2, r3 = indices
            mutant = population[r1] + F * (population[r2] - population[r3])
            mutant = np.clip(mutant, bounds[0], bounds[1])
            # Crossover
            candidate = population[i].copy()
            j_rand = np.random.randint(dimensions)
            for j in range(dimensions):
                if np.random.rand() <= CR or j == j_rand:
                    candidate[j] = mutant[j]
            # Selection
            candidate_value = objective_function(candidate)
            if candidate_value <= values[i]:
                population[i] = candidate
                values[i] = candidate_value
            # Update global best
            if values[i] < global_best_value:
                global_best_position = population[i].copy()
                global_best_value = values[i]
        ssf_values.append(calculate_ssf(population, dimensions))
    return ssf_values

# GA Implementation
def ga(pop_size, dimensions, max_generations, bounds):
    population = initialize_population(pop_size, dimensions, bounds)
    values = np.array([objective_function(ind) for ind in population])
    global_best_idx = np.argmin(values)
    global_best_position = population[global_best_idx].copy()
    global_best_value = values[global_best_idx]
    ssf_values = []

    for generation in range(max_generations):
        # Selection (Tournament)
        new_population = []
        for _ in range(pop_size):
            tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
            tournament_values = values[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_values)]
            new_population.append(population[winner_idx].copy())
        population = np.array(new_population)

        # Crossover
        for i in range(0, pop_size, 2):
            if np.random.rand() < p_c and i + 1 < pop_size:
                point = np.random.randint(1, dimensions)
                parent1, parent2 = population[i].copy(), population[i + 1].copy()
                population[i, point:], population[i + 1, point:] = parent2[point:], parent1[point:]

        # Mutation
        for i in range(pop_size):
            for j in range(dimensions):
                if np.random.rand() < p_m:
                    population[i, j] = np.random.uniform(bounds[0], bounds[1])
        population = np.clip(population, bounds[0], bounds[1])

        # Evaluate
        values = np.array([objective_function(ind) for ind in population])
        best_idx = np.argmin(values)
        if values[best_idx] < global_best_value:
            global_best_position = population[best_idx].copy()
            global_best_value = values[best_idx]
        ssf_values.append(calculate_ssf(population, dimensions))
    return ssf_values

# Random Search with Greedy Selection Implementation
def rs_gs(pop_size, dimensions, max_generations, bounds):
    population = initialize_population(pop_size, dimensions, bounds)
    values = np.array([objective_function(ind) for ind in population])
    global_best_idx = np.argmin(values)
    global_best_position = population[global_best_idx].copy()
    global_best_value = values[global_best_idx]
    ssf_values = []

    for generation in range(max_generations):
        # Generate new random population
        new_population = np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))
        new_values = np.array([objective_function(ind) for ind in new_population])
        # Greedy selection
        for i in range(pop_size):
            if new_values[i] < values[i]:
                population[i] = new_population[i]
                values[i] = new_values[i]
            # Update global best
            if values[i] < global_best_value:
                global_best_position = population[i].copy()
                global_best_value = values[i]
        ssf_values.append(calculate_ssf(population, dimensions))
    return ssf_values

# Run algorithms
ssf_values_pso = pso(pop_size, dimensions, max_generations, bounds)
ssf_values_abc = abc(pop_size, dimensions, max_generations, bounds)
ssf_values_de = de(pop_size, dimensions, max_generations, bounds)
ssf_values_ga = ga(pop_size, dimensions, max_generations, bounds)
ssf_values_rs_gs = rs_gs(pop_size, dimensions, max_generations, bounds)

# Enhanced Plotting
plt.figure(figsize=(10, 6), dpi=100)  # Larger size, high resolution
colors = plt.cm.tab10(np.linspace(0, 1, 5))  # Tableau color palette

# Plot each algorithm's SSF
plt.plot(ssf_values_ga, label="GA", marker='d', markersize=3, linewidth=2, color='orange')
plt.plot(ssf_values_pso, label="PSO", marker='o', markersize=3, linewidth=2, color='green')
plt.plot(ssf_values_de, label="DE", marker='^', markersize=3, linewidth=2, color='blue')
plt.plot(ssf_values_rs_gs, label="RS", marker='*', markersize=3, linewidth=2, color='black')
plt.plot(ssf_values_abc, label="ABC", marker='s', markersize=3, linewidth=2, color='red')

# Customize plot
plt.legend(fontsize=16, loc='upper left')
plt.xlim(0, 100)
plt.ylim(0.2, 1)
plt.xlabel("Iteration", fontsize=18)
plt.ylabel("Signature Factor (\u03B7)", fontsize=18)
# plt.title("SSF Comparison Across Optimization Algorithms", fontsize=20, pad=15)
plt.grid(True, alpha=0.7)
plt.tick_params(axis='both', labelsize=14)

# Add thick border
plt.gca().add_patch(
    patches.Rectangle(
        (0, 0), 1, 1, transform=plt.gca().transAxes,
        linewidth=5, edgecolor='black', facecolor='none'
    )
)

# Ensure tight layout and save
plt.tight_layout()
plt.savefig("output.png", dpi=100, bbox_inches='tight')  # High-res PNG
# plt.savefig("output.svg", format='svg', bbox_inches='tight')  # Vector SVG
plt.show()