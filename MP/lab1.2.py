import numpy as np
import pandas as pd
from scipy import stats

def generate_data_three_factors(beta, x_values, sigma, m):
    """Генерирует данные для модели с тремя факторами."""
    x = np.array(x_values)
    n = len(x)
    
    # Ожидаем 1 коэффициент для константы + 3 для факторов + 3 для их взаимодействий + 1 для взаимодействия всех факторов
    expected_beta_size = 1 + n + (n * (n - 1)) // 2 + 1
    print(f"Количество коэффициентов beta: {len(beta)}, ожидаемое количество: {expected_beta_size}")
    
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
    
    # Добавляем взаимодействие всех факторов
    y += beta[k] * np.prod(x)
    
    # Добавляем случайный шум
    noise = np.random.normal(0, sigma, m)
    return y + noise

def process_data(data):
    """Вычисляет выборочное среднее и дисперсию, а также строит доверительные интервалы."""
    y_mean = np.mean(data)
    y_var = np.var(data, ddof=1)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, len(data)-1)
    margin_mean = t_crit * np.std(data, ddof=1) / np.sqrt(len(data))
    conf_int_mean = (y_mean - margin_mean, y_mean + margin_mean)
    chi2_left = stats.chi2.ppf(alpha/2, len(data)-1)
    chi2_right = stats.chi2.ppf(1 - alpha/2, len(data)-1)
    conf_int_var = ((len(data)-1) * y_var / chi2_right, (len(data)-1) * y_var / chi2_left)
    return y_mean, y_var, conf_int_mean, conf_int_var

def save_to_csv(data, filename):
    """Сохраняет данные в CSV-файл."""
    df = pd.DataFrame({'Y': data})
    df.to_csv(filename, index=False)

# Задание: Модель с тремя факторами
beta_3 = [4, 4, -3, -4, 3, 0, -2, 0]  # 8 коэффициентов: 1 для константы, 3 для факторов, 3 для их взаимодействия, 1 для взаимодействия всех факторов
x_values_3 = [50, 60, 50]  # 3 фактора
sigma_3 = 1.4
m_3 = 10
data_3 = generate_data_three_factors(beta_3, x_values_3, sigma_3, m_3)
y_mean_3, y_var_3, conf_int_mean_3, conf_int_var_3 = process_data(data_3)
save_to_csv(data_3, "data_3.csv")

# Вывод начальной выборки для трех факторов
print("Модель с тремя факторами - начальная выборка:")
df_3 = pd.DataFrame({'X1': [x_values_3[0]]*m_3, 'X2': [x_values_3[1]]*m_3, 'X3': [x_values_3[2]]*m_3, 'Y': data_3})
print(df_3)

print("\nМодель с тремя факторами:")
print("Выборочное среднее:", y_mean_3)
print("Выборочная дисперсия:", y_var_3)
print("Доверительный интервал для среднего:", conf_int_mean_3)
print("Доверительный интервал для дисперсии:", conf_int_var_3)
