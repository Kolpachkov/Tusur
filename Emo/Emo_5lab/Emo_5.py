import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def target_function(x):
    x1, x2 = x
    return 2 * x1**2 + x1 * x2 + x2**2 - 3 * x1


def generate_population(size, dimensions, bounds):
    return [[random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)] for _ in range(size)]


def crossover(parent1, parent2):
    alpha = random.uniform(0, 1)
    return [(alpha * x1 + (1 - alpha) * x2) for x1, x2 in zip(parent1, parent2)]


def mutate(individual, beta, mutation_rate, bounds):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-beta, beta)
            individual[i] = max(min(individual[i], bounds[i][1]), bounds[i][0])
    return individual

def genetic_algorithm(target_function, bounds, population_size, generations, beta, epsilon):
    dimensions = len(bounds)
    population = generate_population(population_size, dimensions, bounds)
    history = []  
    
    for generation in range(generations):
        
        population.sort(key=target_function)
        best_individual = population[0]
        
        
        history.append(population.copy())
        
        
        if target_function(best_individual) <= epsilon:
            break
        
       
        new_population = population[:population_size // 2]  
        while len(new_population) < population_size:
            parents = random.sample(population[:population_size // 2], 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, beta, mutation_rate=0.1, bounds=bounds)
            new_population.append(child)
        
        population = new_population

    return best_individual, target_function(best_individual), history

# Построение графика
def plot_history_3d(history, bounds, target_function, best_solution):
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    x_opt, y_opt = 0.857, -0.429
    z_opt = target_function([ x_opt, y_opt])
    X1, X2 = np.meshgrid(x1, x2)
    Z = target_function([X1, X2])
    
    
    fig = plt.figure(figsize=(12, 8))  
    ax = fig.add_subplot(111, projection='3d')  
    
    
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.set_title('Эволюция популяции в 3D')

    
    for i, population in enumerate(history):
        x_coords = [ind[0] for ind in population]
        y_coords = [ind[1] for ind in population]
        z_coords = [target_function(ind) for ind in population]
        ax.scatter(x_coords, y_coords, z_coords, color='red', s=20, label=f'Поколение {i + 1}' if i == 0 else None)
        for j, (x, y, z) in enumerate(zip(x_coords, y_coords, z_coords)):
            ax.text(x, y, z, f'{i + 1}:{j + 1}', fontsize=8, color='black', ha='center', va='center')

    
    best_x1, best_x2 = best_solution
    best_z = target_function(best_solution)

    ax.scatter(best_x1, best_x2, best_z, color='blue', s=50, label='Лучшее решение', marker='o')
    ax.scatter(x_opt, y_opt, z_opt, color='green', s=50, label='Целевое решение', marker='o')
    ax.text(best_x1, best_x2, best_z, f'Best', fontsize=10, color='blue', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='blue', pad=1))

    plt.legend(loc='upper right')
    plt.show()

bounds = [(-2, 2), (-2, 2)]  # Границы для x1 и x2
population_size = 20  # Количество особей
generations = 1000  # Количество поколений
beta = 0.5  # Параметр мутации
epsilon = 0.01  # Точность решения


best_solution, best_value, history = genetic_algorithm(target_function, bounds, population_size, generations, beta, epsilon)

print("Лучшее решение:", best_solution)
print("Значение целевой функции:", best_value)


plot_history_3d(history, bounds, target_function, best_solution)