import random
import matplotlib.pyplot as plt
import numpy as np
import math

# Длина бинарной хромосомы
L = 5

# Целевая функция, которую нужно минимизировать (пример функции)
def target_function(x):
    return x**4 - 10*x**3 + 36*x**2 + 5*x

# Декодирование бинарной хромосомы в реальное число
def decode(binary_string, a, b):
    r = (2**L - 1)  # Максимальное значение, которое можно представить на L-битах
    value = int(binary_string, 2)  # Преобразование бинарной строки в целое число
    # Декодирование в реальное число из интервала [a, b]
    return a + (b - a) * value / r

# Функция приспособленности (минимизация целевой функции)
def fitness(binary_string, a, b):
    x = decode(binary_string, a, b)
    return target_function(x)

# Селекция: выбор 2 лучших особей на основе приспособленности
def selection(population, a, b):
    return sorted(population, key=lambda ind: fitness(ind, a, b))[:2]

# Кроссинговер: одноточечный кроссинговер между двумя бинарными строками
def crossover(parent1, parent2):
    point = random.randint(1, L - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Мутация: инверсия случайного бита в бинарной строке с вероятностью 1/L
def mutate(child):
    mutation_prob = 1 / L
    child_list = list(child)
    for i in range(L):
        if random.random() < mutation_prob:
            child_list[i] = '1' if child_list[i] == '0' else '0'
    return ''.join(child_list)

# Основной генетический алгоритм
def genetic_algorithm(a, b, K, N, epsilon):
    # Шаг 1: Инициализация популяции случайными бинарными строками
    population = [''.join(random.choice('01') for _ in range(L)) for _ in range(K)]
    
    for generation in range(N):
        # Шаг 2: Оценка приспособленности и выбор лучших особей
        best_individuals = selection(population, a, b)
        
        # Шаг 3: Создание новой популяции с помощью кроссинговера и мутации
        new_population = []
        while len(new_population) < K:
            parent1, parent2 = random.sample(best_individuals, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        # Шаг 4: Замена старой популяции
        population = new_population[:K]
        
        # Шаг 5: Оценка лучшей особи
        best_solution = min(population, key=lambda ind: fitness(ind, a, b))
        best_value = fitness(best_solution, a, b)
        
        # Преобразование бинарного решения обратно в реальное значение
        best_real_value = decode(best_solution, a, b)
        
        # Проверка критерия завершения
        if best_value <= epsilon:
            break

    return best_real_value, best_value

# Построение графика целевой функции
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

# Пример использования
a = -7  # Левая граница интервала
b = 7   # Правая граница интервала
K = 50  # Размер популяции
N = 100  # Количество поколений
epsilon = 0.0001  # Порог точности

best_x, best_f = genetic_algorithm(a, b, K, N, epsilon)
print(f"Лучшее решение: x = {best_x}, f(x) = {best_f}")
plt.scatter(best_x, best_f, color='red', label=f"Точка минимума({best_x}, {best_f})")
plot_function(a, b)
