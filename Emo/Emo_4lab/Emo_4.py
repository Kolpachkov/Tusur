import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def target_function(x):
    return x**4 - 10 * (x**3) + 36 * x**2 + 5 * x


def decode(real_value, a, b):
    return a + (b - a) * real_value  


def fitness(real_value, a, b):
    x = decode(real_value, a, b)
    return target_function(x)


def selection(population, a, b):
    return sorted(population, key=lambda ind: fitness(ind, a, b))[:2]


def crossover(parent1, parent2):
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

def mutate(child, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_value = random.uniform(-beta, beta)  
        child += mutation_value
        child = max(min(child, 1), 0)  
    return child

def genetic_algorithm(a, b, K, N, epsilon):
    
    population = [random.uniform(0, 1) for _ in range(K)]  # Инициализация вещественной популяции в пределах [0, 1]
    initial_population = population.copy()
    best_values = []

    for generation in range(N):
        best_individuals = selection(population, a, b)
        new_population = []

        while len(new_population) < K:
            parent1, parent2 = random.sample(best_individuals, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = new_population[:K]

        best_solution = min(population, key=lambda ind: fitness(ind, a, b))
        best_value = fitness(best_solution, a, b)
        best_real_value = decode(best_solution, a, b)

        best_values.append(best_value)

        if abs(best_value) <= epsilon:
            break

    return best_real_value, best_value, best_values, initial_population


def save_to_csv(a, b, K, N, epsilon, initial_population, best_x, best_f):
    file_exists = os.path.isfile('genetic_algorithm_results.csv')

    with open('genetic_algorithm_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['a', 'b', 'K', 'N', 'epsilon', 'Best x', 'Best f(x)', 'Initial Population'])

        writer.writerow([a, b, K, N, epsilon, best_x, best_f, initial_population])


def plot_function(a, b):
    x = np.linspace(a, b, 400)
    y = target_function(x)
    plt.plot(x, y, label="f(x)")
    plt.title("Целевая функция")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


a = -7  
b = 7  
K = 50  
N = 500 
alpha = 1
beta = 0.5
epsilon = 0.001  

best_x, best_f, best_values, initial_population = genetic_algorithm(a, b, K, N, epsilon)

save_to_csv(a, b, K, N, epsilon, initial_population, best_x, best_f)

print(f"Лучшее решение: x = {best_x}, f(x) = {best_f}")

plt.scatter(best_x, best_f, color='red', label=f"Точка минимума({best_x}, {best_f})")
plot_function(a, b)
