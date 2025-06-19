import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy.signal import find_peaks
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

# --- ПАРАМЕТР К ---
K = 45  # Из таблицы вариантов

# --- ПЕРЕДАТОЧНЫЕ ФУНКЦИИ ---
W1 = ctrl.TransferFunction([K], [1])
W2 = ctrl.TransferFunction([0.1], [0.001, 1])
W3 = ctrl.TransferFunction([10], [0.04, 1])
W4 = ctrl.TransferFunction([1], [0.004, 0.376])
W5 = ctrl.TransferFunction([0.8], [1])
W6 = ctrl.TransferFunction([1], [0.065, 0])  # Интегратор
W7 = ctrl.TransferFunction([0.8], [1])
W9 = ctrl.TransferFunction([0.14], [1])  # Только W9 в обратной связи!

# --- СТРУКТУРНАЯ СХЕМА ---
W_total = W1 * W2 * W3 * W4 * W5 * W6 * W7
W_closed = ctrl.feedback(W_total, W9)

print("\n=== Передаточная функция замкнутой САУ ===")
print(W_closed)
print("Коэффициенты знаменателя:", W_closed.den[0][0])  # Проверим свободный член

# --- ПЕРЕХОДНАЯ ХАРАКТЕРИСТИКА ---
t, y = ctrl.step_response(W_closed)
y_final = y[-1]

# --- ПРЯМЫЕ ОЦЕНКИ ---
def settling_time(t, y, tol=0.05):
    for i in range(len(y) - 1, -1, -1):
        if abs(y[i] - y_final) > tol * abs(y_final):
            if i + 1 < len(t):
                return t[i + 1]
            else:
                return t[i]
    return t[0]

def overshoot(y):
    return (max(y) - y_final) / y_final * 100

def rise_time(t, y):
    for i in range(len(y)):
        if y[i] >= y_final:
            return t[i]
    return t[-1]

def time_peak(t, y):
    return t[np.argmax(y)]

def oscillation_freq(t, y):
    peaks, _ = find_peaks(y)
    if len(peaks) >= 2:
        T0 = t[peaks[1]] - t[peaks[0]]
        return 2 * np.pi / T0
    return 0

def count_oscillations(t, y, treg):
    peaks, _ = find_peaks(y)
    return sum(t[p] <= treg for p in peaks)

treg = settling_time(t, y)
Hm = overshoot(y)
tn = rise_time(t, y)
tm = time_peak(t, y)
w = oscillation_freq(t, y)
N = count_oscillations(t, y, treg)

# --- КОРНЕВЫЕ ОЦЕНКИ ---
poles = ctrl.poles(W_closed)

stable_poles = [p for p in poles if p.real < 0]

if stable_poles:
    alpha_min = min(abs(p.real) for p in stable_poles)
    t_est = 3 / alpha_min
else:
    alpha_min = 0
    t_est = float('inf')

mu_candidates = [abs(p.imag) / abs(p.real) for p in stable_poles if p.imag != 0]
mu = max(mu_candidates) if mu_candidates else 0

# --- ЧАСТОТНЫЕ ОЦЕНКИ ---
W_open = ctrl.series(W_total, W9)  # Правильное разомкнутое звено
gm, pm, sm, wg, wp, ws = ctrl.stability_margins(W_open, returnall=True)

# --- ВЫВОД ---
print(f"\n=== Прямые оценки ===")
print(f"Время регулирования tрег: {treg:.2f} с")
print(f"Перерегулирование Hm: {Hm:.2f} %")
print(f"Время нарастания tн: {tn:.2f} с")
print(f"Время первого максимума tm: {tm:.2f} с")
print(f"Частота колебаний ω: {w:.2f} рад/с")
print(f"Число колебаний N: {N}")

print(f"\n=== Корневые оценки ===")
print(f"Степень устойчивости αmin: {alpha_min:.2f}")
print(f"Приближенное tрег ≈ 3 / αmin = {t_est:.2f} с")
print(f"Колебательность μ: {mu:.2f}")

print(f"\n=== Частотные оценки ===")
if len(pm) > 0 and len(wp) > 0:
    print(f"Запас по фазе Δφ: {pm[0]:.2f}° при ω = {wp[0]:.2f} рад/с")
if len(gm) > 0 and len(wg) > 0:
    print(f"Запас по модулю ΔL: {20 * np.log10(gm[0]):.2f} дБ при ω = {wg[0]:.2f} рад/с")

# --- ГРАФИК ПЕРЕХОДНОЙ ХАРАКТЕРИСТИКИ ---
plt.figure()
plt.plot(t, y, label="y(t)")
plt.axhline(y_final, color='gray', linestyle='--', label='y конечное')
plt.axhline(y_final * 1.05, color='red', linestyle='--', alpha=0.5, label='+5% зона')
plt.axhline(y_final * 0.95, color='red', linestyle='--', alpha=0.5, label='-5% зона')
plt.axvline(treg, color='orange', linestyle=':', label=f"tрег ≈ {treg:.2f} c")
plt.axvline(tm, color='purple', linestyle=':', label=f"tm ≈ {tm:.2f} c")
plt.axvline(tn, color='green', linestyle=':', label=f"tн ≈ {tn:.2f} c")
plt.title("Переходная характеристика y(t)")
plt.xlabel("Время (с)")
plt.ylabel("Выход")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
