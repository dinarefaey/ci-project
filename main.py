import random
import numpy as np


# objective function to minimize cost of assinment with the most required task
def objective_function(x, task_requirements):
    # x is the matrix for resources and tasks , where i in matrix represent resources and j represent tasks
    # x[i,j] = 1 then resource i is assigned to task j, and 0 otherwise.
    # let the cost function where the cost of assigning a resource to a task is the product of their indices,
    # add a bonus to the cost of assigning a resource to a task that has a high requirement.
    cost = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 1:
                cost += i * j
                cost += 10 * task_requirements[j]
    return cost


resourses = int(input("enter number of resources"))
tasks = int(input("enter number of tasks"))
bounds = [(0, 1)] * resourses * tasks
bounds = np.array(bounds)
# input the number of times each task needs to be assigned)
task_requirements = []
# Take input from user n times and append to the list of task requirments
for i in range(tasks):
    element = int(input("Enter number of time to assgin task {}: ".format(i + 1)))
    task_requirements.append(element)
# Set the population size and number of iterations for the DE algorithm
population_size = int(input("Enter population size for DE algorithm"))
num_iterations = int(input("Enter number of iteration for DE algorithm "))


# Define the DE algorithm
def differential_evolution(objective_function, bounds, task_requirements, population_size, num_iterations, F=0.5,
                           CR=0.7):
    # Initialize the population
    population = np.zeros((population_size, bounds.shape[0]))
    for i in range(population_size):
        population[i, :] = np.random.uniform(bounds[:, 0], bounds[:, 1])

    # Iterate over the specified number of generations
    for i in range(num_iterations):
        # Iterate over each individual in the population
        for j in range(population_size):
            # Select three distinct individuals at random
            indices = list(range(population_size))
            indices.remove(j)
            a, b, c = random.sample(indices, 3)

            # Generate a new candidate solution by combining the information from the three individuals
            mutant = population[a, :] + F * (population[b, :] - population[c, :])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            # Select a random index and create a new candidate solution by recombining
            # the information from the mutant and target individuals
            index = random.randint(0, bounds.shape[0] - 1)
            trial = np.zeros(bounds.shape[0])
            for k in range(bounds.shape[0]):
                if random.random() < CR or k == index:
                    trial[k] = mutant[k]
                else:
                    trial[k] = population[j, k]

            # Evaluate the fitness of the trial solution and update the population
            # if it is better than the target individual
            trial_fitness = objective_function(trial.reshape((resourses, tasks)), task_requirements)
            if trial_fitness < objective_function(population[j, :].reshape((resourses, tasks)), task_requirements):
                population[j, :] = trial

    # Return the best individual found
    best_individual = population[np.argmin(
        [objective_function(individual.reshape((resourses, tasks)), task_requirements) for individual in population]),
                      :]
    return best_individual


# Run the differential evolution algorithm
best_assignment = differential_evolution(objective_function, bounds, task_requirements, population_size, num_iterations)

# Print the best assignment found
best_assignment = np.round(best_assignment.reshape(resourses, tasks))
print("Best assignment found: \n", best_assignment)