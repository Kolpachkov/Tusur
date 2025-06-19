from sympy import symbols, Matrix, simplify, solve, expand
import numpy as np
from control import tf, feedback, step_response, bode_plot, nyquist_plot, margin, poles
import matplotlib.pyplot as plt
import json

# --- Исходные параметры ---
U_эму = 230
I_вх = 0.01
r_вх = 2100
L_вх = 100
r1 = 3.35
L1 = 0.6
J_дв = 0.065
R_я = 0.376
L_я = 0.004

K_эму = U_эму / (I_вх * r_вх)
T_эму = L1 / r1
T_вх = L_вх / r_вх
K_ос = 0.12

# --- Передаточные функции ---
W_эму = tf([K_эму], [T_эму * T_вх, T_эму + T_вх, 1])
W_двигатель = tf([1.24], [0.0076, 0.4, 1])
W_разомкнутая = W_эму * W_двигатель * K_ос
W_замкнутая = feedback(W_разомкнутая, 1)

# --- Устойчивость ---
полюса = poles(W_замкнутая)
Re = [p.real for p in полюса]
Im = [p.imag for p in полюса]

plt.figure()
plt.scatter(Re, Im, color='red', label='Полюса')
plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Мнимая ось')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Полюса замкнутой САУ")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

устойчива = all(np.real(p) < 0 for p in полюса)

# --- Переходная характеристика ---
t, y = step_response(W_замкнутая)
plt.plot(t, y)
plt.title("Переходная характеристика")
plt.xlabel("Время, с")
plt.ylabel("Выход")
plt.grid()
plt.show()

# --- Критерий Найквиста ---
полюса_разомк = poles(W_разомкнутая)
Re_open = [p.real for p in полюса_разомк]
Im_open = [p.imag for p in полюса_разомк]

nyquist_plot(W_разомкнутая)
plt.scatter(Re_open, Im_open, color='red', marker='x', label='Полюса разомкнутой САУ')
plt.title("Критерий Найквиста")
plt.grid()
plt.legend()
plt.show()

# --- Частотные характеристики ---
mag, phase, omega = bode_plot(W_разомкнутая, dB=True, plot=True)
gm, pm, wg, wp = margin(W_разомкнутая)

# --- Годограф Михайлова (исправленный) ---
den_mikh = W_замкнутая.den[0][0]
omega_m = np.logspace(-1, 2, 1000)
H = []

for w in omega_m:
    jw = 1j * w
    val = sum(coef * (jw ** i) for i, coef in enumerate(reversed(den_mikh)))
    H.append(val)

H = np.array(H)
phi = np.angle(H)
amp = np.abs(H)

# Полярный график
plt.figure()
ax = plt.subplot(1, 1, 1, projection='polar')
ax.plot(phi, amp, label="Годограф Михайлова")
ax.set_title("Годограф Михайлова")
ax.grid(True)
plt.legend()
plt.show()

# Декартовы координаты
x = H.real
y = H.imag

plt.figure()
plt.plot(x, y, label="Годограф Михайлова (декартовые координаты)")
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Годограф Михайлова (декартовые координаты)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# --- Критерий Рауса-Гурвица ---
num_raz = np.array(W_разомкнутая.num[0][0])
den_raz = np.array(W_разомкнутая.den[0][0])
max_len = max(len(den_raz), len(num_raz))
num_raz = np.pad(num_raz, (max_len - len(num_raz), 0))
den_raz = np.pad(den_raz, (max_len - len(den_raz), 0))

s, Koc = symbols('s Koc')
D_poly = sum(den_raz[i] * s**(max_len - 1 - i) for i in range(max_len))
N_poly = sum(num_raz[i] * s**(max_len - 1 - i) for i in range(max_len))
char_poly = expand(D_poly + Koc * N_poly)
coeffs = char_poly.as_poly(s).all_coeffs()
a0, a1, a2, a3, a4 = coeffs

H = Matrix([
    [a1, a3, 0],
    [a0, a2, a4],
    [0, a1, a3]
])

D = simplify(H.det())
roots = solve(D, Koc)

print("Границы устойчивости Koc:", roots)

# Проверка корней замкнутой системы при границах
from sympy import lambdify

char_poly_func = lambdify((s, Koc), char_poly, 'numpy')

for val in roots:
    k_val = float(val.evalf())
    # Подставляем Koc = k_val и получаем численные коэффициенты
    char_num = []
    for i in range(len(coeffs)):
        c = coeffs[i].subs(Koc, k_val)
        char_num.append(float(c.evalf()))
    # char_num идут от старшей степени s^4 к младшей s^0
    char_num = np.array(char_num)

    # Корни полинома
    roots_poly = np.roots(char_num)
    print(f"\nПри Koc = {k_val}:")
    print("Корни характеристического полинома:", roots_poly)
    print("Действительная часть корней:", roots_poly.real)
