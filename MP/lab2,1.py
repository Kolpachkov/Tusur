import numpy as np
import pandas as pd
import scipy.stats as stats

# Коэффициенты регрессии
beta_0 = 4
beta_1 = 4
beta_2 = -3
beta_3 = -4
beta_12 = 3
beta_13 = 0
beta_23 = -2
beta_123 = 0

# Центр планирования и шаги варьирования
X0_1, X0_2, X0_3 = 50, 60, 50
delta_X1, delta_X2, delta_X3 = 30, 30, 30

# Кодированные уровни факторов (-1 и 1)
X_coded = np.array([
    [-1, -1, -1],
    [-1, -1,  1],
    [-1,  1, -1],
    [-1,  1,  1],
    [ 1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1,  1]
])

# Фиктивная переменная (всегда 1)
X0 = np.ones(len(X_coded))

# Преобразование в натуральные значения
X1 = X0_1 + X_coded[:, 0] * delta_X1
X2 = X0_2 + X_coded[:, 1] * delta_X2
X3 = X0_3 + X_coded[:, 2] * delta_X3

# Генерация выходных значений с шумом N(0, 1)
m = 3  # Количество повторов эксперимента
sigma = 1.4  # Дисперсия шума

def response(x1, x2, x3):
    noise = np.random.normal(0, sigma, m)
    return (beta_0 + beta_1 * x1 + beta_2 * x2 + beta_3 * x3 +
            beta_12 * x1 * x2 + beta_13 * x1 * x3 + beta_23 * x2 * x3 + beta_123 * x1 * x2 * x3 + noise)

# Рассчет отклика
Y = np.array([response(x1, x2, x3) for x1, x2, x3 in zip(X1, X2, X3)])

# Создание таблицы эксперимента
df = pd.DataFrame({
    "Фиктивная переменная": X0,
    "X1 (код.)": X_coded[:, 0],
    "X2 (код.)": X_coded[:, 1],
    "X3 (код.)": X_coded[:, 2],
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
X_extended = np.column_stack((np.ones(len(X_coded)), X_coded, X_coded[:, 0] * X_coded[:, 1],
                              X_coded[:, 0] * X_coded[:, 2], X_coded[:, 1] * X_coded[:, 2],
                              X_coded[:, 0] * X_coded[:, 1] * X_coded[:, 2]))
B = np.linalg.lstsq(X_extended, Y_mean, rcond=None)[0]
print("\nКоэффициенты регрессии:", B)

# Проверка значимости коэффициентов (t-критерий Стьюдента)
S_B = np.sqrt(S2.mean() / (len(X1) * m))
t = np.abs(B) / S_B
t_crit = stats.t.ppf(1 - 0.025, df=f1 * f2)
print(f"\nЗначимость коэффициентов (t-критерий): {t} {t_crit}")
print(f"\nЗначимость коэффициентов (t-критерий): {t > 2.3}")

# Проверка адекватности модели (F-критерий Фишера)
mean_of_squared_diff = ((Y - Y_mean[:, np.newaxis]) ** 2).mean()
Fp = mean_of_squared_diff / S2.mean() if S2.mean() != 0 else 0
F_crit = stats.f.ppf(1 - 0.05, len(X1) - len(B), f1 * f2)

print(f"\nАдекватность модели (F-критерий): {Fp < 3.4}")
print(f"\nАдекватность модели (F-критерий): {Fp}")