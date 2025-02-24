import numpy as np
import pandas as pd
from scipy import stats

def generate_data_2(beta, x_values, sigma, m,noise):
    """Генерирует данные для модели объекта исследования."""
    x = np.array(x_values)
    n = len(x)
    expected_beta_size = 1 + n + 1  # Ожидаем 4 коэффициента: 1 для константы, 2 для факторов, 1 для их взаимодействия
    
    if len(beta) != expected_beta_size:
        raise ValueError(f"Некорректное количество коэффициентов beta: {len(beta)}, ожидалось {expected_beta_size}")
    
    y = np.full(m, beta[0])  # Начальное значение (для константы)
    for i in range(n):
        y += beta[i + 1] * x[i]  # Добавляем линейные компоненты
    
    # Добавляем взаимодействие факторов
    y += beta[n + 1] * x[0] * x[1]
    
    # Добавляем случайный шум
    
    return y + noise

def generate_data_three_factors(beta, x_values, sigma, m,noise):
    """Генерирует данные для модели с тремя факторами."""
    x = np.array(x_values)
    n = len(x)
    
    # Ожидаем 1 коэффициент для константы + 3 для факторов + 3 для их взаимодействий + 1 для взаимодействия всех факторов
    expected_beta_size = 1 + n + (n * (n - 1)) // 2 + 1
    
    if len(beta) != expected_beta_size:
        raise ValueError(f"Некорректное количество коэффициентов beta: {len(beta)}, ожидалось {expected_beta_size}")
    
    y = np.full(m, beta[0], dtype=np.float64)  # Указываем тип данных для y как float64
    for i in range(n):
        y += beta[i + 1] * x[i]  # Добавляем линейные компоненты
    
    # Добавляем взаимодействие факторов
    k = n + 1
    for i in range(n):
        for j in range(i + 1, n):
            y += beta[k] * x[i] * x[j]
            k += 1
    y += beta[k] * np.prod(x)

    return y + noise

def confidence_interval_for_variance(data, alpha=0.03):
    # Выборочная дисперсия
    s2 = np.var(data, ddof=1)
    
    # Количество элементов в выборке
    n = len(data)
    
    # Степени свободы
    df = n - 1
    
    # Критические значения для распределения χ2
    chi2_lower = stats.chi2.ppf(alpha / 2, df)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
    
    # Доверительный интервал для дисперсии
    lower_bound = (df * s2) / chi2_upper
    upper_bound = (df * s2) / chi2_lower
    
    return lower_bound, upper_bound

# Пример данных
data = np.random.normal(0, 1, 10)

def process_data(data):
    """Вычисляет выборочное среднее и дисперсию, а также строит доверительные интервалы."""
    y_mean = np.mean(data)
    y_var = np.var(data, ddof=1)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, len(data)-1)
    print("\nT-crit:", t_crit)
    margin_mean = t_crit * np.std(data, ddof=1) / np.sqrt(len(data))
    conf_int_mean = (y_mean - margin_mean, y_mean + margin_mean)
    chi2_left = 3.325
    chi2_right = 16.919
    print("X  left:", chi2_left)
    print("X  Right:", chi2_right)
    conf_int_var = ((len(data)-1) * y_var / chi2_right, (len(data)-1) * y_var / chi2_left)
    return y_mean, y_var, conf_int_mean, conf_int_var

def save_to_csv(data, filename):
    """Сохраняет данные в CSV-файл."""
    df = pd.DataFrame({'Y': data})
    df.to_csv(filename, index=False)


sigma= 1.5
m = 10
noise = np.random.normal(0, sigma, m)


# Задание 1 и 2: Модель с двумя факторами
beta_2 = [4, 4, -3, 3]  # 4 коэффициента: 1 для константы, 2 для факторов, 1 для их взаимодействия
x_values_2 = [50, 60]

data_2 = generate_data_2(beta_2, x_values_2, sigma, m, noise)
y_mean_2, y_var_2, conf_int_mean_2, conf_int_var_2 = process_data(data_2)
save_to_csv(data_2, "data_2.csv")


# Вывод начальной выборки


print("\nМодель с двумя факторами:")
print("Выборочное среднее:", y_mean_2)
print("Выборочная дисперсия:", y_var_2)
print("Доверительный интервал для среднего:", conf_int_mean_2)
print("Доверительный интервал для дисперсии:", conf_int_var_2)

beta_3 = [4, 4, -3, -4, 3, 0, -2, 0]  
x_values_3 = [50, 60, 50]  # 3 фактора

data_3 = generate_data_three_factors(beta_3, x_values_3, sigma, m, noise)
y_mean_3, y_var_3, conf_int_mean_3, conf_int_var_3 = process_data(data_3)
save_to_csv(data_3, "data_3.csv")



print("\nМодель с тремя факторами:")
print("Выборочное среднее:", y_mean_3)
print("Выборочная дисперсия:", y_var_3)
print("Доверительный интервал для среднего:", conf_int_mean_3)
print("Доверительный интервал для дисперсии:", conf_int_var_3)