import numpy as np
import pandas as pd
import scipy.stats as stats

# Коэффициенты регрессии
beta_0 = 4
beta_1 = 4
beta_2 = -3
beta_12 = 3

# Центр планирования и шаги варьирования
X0_1, X0_2 = 50, 60
delta_X1, delta_X2 = 30, 30

# Кодированные уровни факторов (-1 и 1)
X_coded = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

# Фиктивная переменная (всегда 1)
X0 = np.ones(len(X_coded))

# Преобразование в натуральные значения
X1 = X0_1 + X_coded[:, 0] * delta_X1
X2 = X0_2 + X_coded[:, 1] * delta_X2 

# Генерация выходных значений с шумом N(0, 1)
m = 3  # Количество повторов эксперимента
sigma = 1.4  # Дисперсия шума

def response(x1, x2):
    noise = np.random.normal(0, sigma, m)
    return beta_0 + beta_1 * x1 + beta_2 * x2 + beta_12 * x1 * x2 + noise

def response2(x1, x2, beta0, beta1, beta2, beta3):
    noise = np.random.normal(0, sigma, m)
    return beta0 + beta1 * x1 + beta2 * x2 + beta3 * x1 * x2 + noise

# Рассчет отклика
Y = np.array([response(x1, x2) for x1, x2 in zip(X1, X2)])

# Создание таблицы эксперимента
df = pd.DataFrame({
    "Фиктивная переменная": X0,
    "X1 (код.)": X_coded[:, 0],
    "X2 (код.)": X_coded[:, 1],
    "Y1": Y[:, 0],
    "Y2": Y[:, 1],
    "Y3": Y[:, 2]
})

print("Таблица эксперимента:\n", df)

# Проверка воспроизводимости (дисперсия однородности)
Y_mean = Y.mean(axis=1)
S2 = Y.var(axis=1, ddof=1)
Gp = S2.max() / S2.sum()  # Критерий Кохрена
f1, f2 = m - 1, len(X1)
Gt = stats.f.ppf(0.95, f1, (f2 - 1) * f1) / ((stats.f.ppf(0.95, f1, (f2 - 1) * f1)) + (f2 - 1) * f1)

print(f"\nКритерий Кохрена Gp: {Gp}, Gt: {Gt}, Однородность: {Gp < Gt}")

# Расчет коэффициентов регрессии
X_extended = np.column_stack((np.ones(len(X_coded)), X_coded, X_coded[:, 0] * X_coded[:, 1]))
B = np.linalg.lstsq(X_extended, Y_mean, rcond=None)[0]
print("\nКоэффициенты регрессии:", B)

# Прогнозирование отклика на основе закодированных значений
Y2 = np.array([response2(x1_coded, x2_coded, B[0], B[1], B[2], B[3]) for x1_coded, x2_coded in zip(X_coded[:, 0], X_coded[:, 1])])

# Создание таблицы с рассчитанными откликами
df2 = pd.DataFrame({
    "Фиктивная переменная": X0,
    "X1 (код.)": X_coded[:, 0],
    "X2 (код.)": X_coded[:, 1],
    "Y1": Y2[:, 0],
    "Y2": Y2[:, 1],
    "Y3": Y2[:, 2]
})
print("Таблица эксперимента с прогнозируемыми откликами:\n", df2)

# Разница между Y и Y2
Y_diff = Y - Y2
Y_diff_mean = Y_diff.mean(axis=1)

Y_diff_squared = Y_diff_mean ** 2
mean_of_squared_diff = Y_diff_squared.mean()
print("\nСреднее значение квадратов разницы:\n", mean_of_squared_diff)


# Проверка значимости коэффициентов (t-критерий Стьюдента)
S_B = np.sqrt(S2.mean() / (len(X1) * m))
t = np.abs(B) / S_B
t_crit = stats.t.ppf(1 - 0.025, df=f1 * f2)
print(f"\nЗначимость коэффициентов (t-критерий): {t} {t_crit}")
print(f"\nЗначимость коэффициентов (t-критерий): {t > t_crit}")

# Проверка адекватности модели (F-критерий Фишера)

Fp = mean_of_squared_diff / S2.mean() if S2.mean() != 0 else 0  # Проверка на нулевую дисперсию
F_crit = stats.f.ppf(1 - 0.05, len(X1) - len(B), f1 * f2)

print(f"\nАдекватность модели (F-критерий): {Fp < 5.32}")
print(f"\nАдекватность модели (F-критерий): {Fp }")
