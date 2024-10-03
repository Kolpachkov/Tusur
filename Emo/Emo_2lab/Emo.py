import random
import matplotlib.pyplot as plt
import numpy as np
import math

def target_function(x):
    return x**4 - 10*x**3 + 36*x**2 + 5*x
def genetic_algorithm(a, b, K, N, epsilon):
    population = [random.randint(a, b) for _ in range(K)]
    
    
    def fitness(x):
        return target_function(x)
    
    # Функция отбора лучших (выбираем 2 лучших по значению функции)
    def selection(population):
        return sorted(population, key=fitness)[:2]
    
    # Функция кроссинговера (скрещивание)
    def crossover(parent1, parent2):
        # Среднее значение между родителями с небольшим случайным отклонением
        return (parent1 + parent2) // 2 + random.randint(-1, 1)
    
    # Функция мутации (случайное небольшое изменение)
    def mutate(child):
        return child + random.randint(-3, 3)
    
    # Основной цикл алгоритма
    for generation in range(N):
        
        best_individuals = selection(population)
        
        # Кроссинговер и мутации для создания новой популяции
        new_population = []
        for _ in range(K):
            parent1, parent2 = random.sample(best_individuals, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            # Ограничение на диапазон значений
            child = max(a, min(b, child))
            new_population.append(child)
        
        # Замена популяции
        population = new_population
        
        # Лучшая особь текущего поколения
        best_solution = min(population, key=fitness)
        best_value = target_function(best_solution)
        
        # Проверка критерия точности
        if best_value <= epsilon:
            break

    # Возвращаем лучшую найденную особь и её значение функции
    return best_solution, best_value
def plot_function(a, b):
    x = np.linspace(a, b, 400)
    y = target_function(x)
    
    plt.plot(x, y, label="f(x)")
    plt.title("График целевой функции")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()
# Пример использования
a = -7  # Левая граница
b = 7  # Правая граница
K = 50   # Количество особей
N = 100  # Количество поколений
epsilon = 0.0001  # Точность

best_x, best_f = genetic_algorithm(a, b, K, N, epsilon)
print(f"Лучшее решение: x = {best_x}, f(x) = {best_f}")
plt.scatter(best_x, best_f, color='red', label=f"Точка минимума({best_x}, {best_f})")
plot_function(a, b)