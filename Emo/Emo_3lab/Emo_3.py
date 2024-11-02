import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

L = 16  # Длина строки


def target_function(x):
    return x**4 - 10*(x**3) + 36 * x**2 + 5 * x


def binary_to_gray(binary_string):
    binary_value = int(binary_string, 2)
    gray_value = binary_value ^ (binary_value >> 1)
    return f'{gray_value:0{L}b}'  


def gray_to_binary(gray_string):
    gray_value = int(gray_string, 2)
    binary_value = gray_value
    shift = 1
    while gray_value >> shift:
        binary_value ^= gray_value >> shift
        shift += 1
    return f'{binary_value:0{L}b}'


def decode(gray_string, a, b):
    binary_string = gray_to_binary(gray_string)
    r = (2**L - 1)
    value = int(binary_string, 2)
    return a + (b - a) * value / r


def fitness(gray_string, a, b):
    x = decode(gray_string, a, b)
    return target_function(x)


def selection(population, a, b):
    return sorted(population, key=lambda ind: fitness(ind, a, b))[:2]


def crossover(parent1, parent2):
    point = random.randint(1, L - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(child):
    mutation_prob = 1 / L  # Вероятность мутации
    child_list = list(child)
    for i in range(L):
        if random.random() < mutation_prob:
            child_list[i] = '1' if child_list[i] == '0' else '0'
    return ''.join(child_list)


def genetic_algorithm(a, b, K, N, epsilon):
    
    population = [binary_to_gray(''.join(random.choice('01') for _ in range(L))) for _ in range(K)]

   
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
K = 20  
N = 15  
epsilon = 0.001  


best_x, best_f, best_values, initial_population = genetic_algorithm(a, b, K, N, epsilon)


save_to_csv(a, b, K, N, epsilon, initial_population, best_x, best_f)

print(f"Лучшее решение: x = {best_x}, f(x) = {best_f}")


plt.scatter(best_x, best_f, color='red', label=f"Точка минимума({best_x}, {best_f})")
plot_function(a, b)
