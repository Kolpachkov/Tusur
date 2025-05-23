import numpy as np
import matplotlib.pyplot as plt
import json
from control import tf, feedback, step_response, bode_plot, nyquist_plot, margin, poles
from sympy import symbols, Matrix, simplify, solve, expand
from scipy.signal import freqs

# --- Паспортные данные и расчёт параметров ---
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

# --- Критерий Найквиста с полюсами разомкнутой САУ ---
полюса_разомк = poles(W_разомкнутая)
Re_open = [p.real for p in полюса_разомк]
Im_open = [p.imag for p in полюса_разомк]

nyquist_plot(W_разомкнутая)
plt.scatter(Re_open, Im_open, color='red', marker='x', label='Полюса разомкнутой САУ')
plt.title("Критерий Найквиста")
plt.grid()
plt.legend()
plt.show()

# --- Частотные характеристики и запасы устойчивости ---
mag, phase, omega = bode_plot(W_разомкнутая, dB=True, plot=True)
gm, pm, wg, wp = margin(W_разомкнутая)

# --- Годограф Михайлова (полярный) ---
den_mikh = W_замкнутая.den[0][0]
omega_m = np.logspace(-1, 2, 1000)
_, H = freqs([1], den_mikh, worN=omega_m)
phi = np.angle(H)
amp = np.abs(H)

plt.figure()
ax = plt.subplot(1, 1, 1, projection='polar')
ax.plot(phi, amp, label="Годограф Михайлова")
ax.set_title("Годограф Михайлова")
ax.grid(True)
plt.legend()
plt.show()

x = amp * np.cos(phi)
y = amp * np.sin(phi)

plt.figure()
plt.plot(x, y, label="Годограф Михайлова (декартовы координаты)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Годограф Михайлова (декартовые координаты)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

# --- Критерий Рауса-Гурвица ---
W_база = W_эму * W_двигатель
num = [K_эму * 1.24]  # = 13.64

num_raz = W_разомкнутая.num[0][0]
den_raz = W_разомкнутая.den[0][0]
den = np.polyadd(den_raz, num_raz)

print("Вычисленные коэффициенты знаменателя замкнутой системы:", den)

s, Kос_симв = symbols('s Kос')
D_poly = sum(den[i] * s**(len(den) - 1 - i) for i in range(len(den)))
N_poly = num[0]
хар_полином = expand(D_poly + Kос_симв * N_poly)
coeffs = хар_полином.as_poly(s).all_coeffs()
a0, a1, a2, a3 = coeffs[:4]
H = Matrix([
    [a1, a3, 0],
    [a0, a2, 0],
    [0, a1, a3]
])
D = simplify(H.det())
решения = solve(D, Kос_симв)

# --- Запись в JSON ---
результаты = {
    "параметры": {
        "U_эму": U_эму,
        "I_вх": I_вх,
        "r_вх": r_вх,
        "L_вх": L_вх,
        "r1": r1,
        "L1": L1,
        "K_эму": round(K_эму, 4),
        "T_эму": round(T_эму, 4),
        "T_вх": round(T_вх, 4),
        "K_ос": K_ос
    },
    "передаточные_функции": {
        "W_эму": str(W_эму),
        "W_двигатель": str(W_двигатель),
        "W_разомкнутая": str(W_разомкнутая),
        "W_замкнутая": str(W_замкнутая)
    },
    "устойчивость": {
        "полюса": [str(p) for p in полюса],
        "устойчива": устойчива
    },
    "частотные_характеристики": {
        "запас_по_усилению_dB": round(20 * np.log10(gm), 2) if gm > 0 else None,
        "запас_по_фазе_град": round(pm, 2),
        "частота_пересечения_усиления": round(wg, 2),
        "частота_пересечения_фазы": round(wp, 2)
    },
    "раус_гурвиц": {
        "характеристический_полином": str(хар_полином),
        "определитель_Гурвица": str(D),
        "границы_Kос": [str(sol) for sol in решения]
    },
    "формулы": {
        "K_эму": "U_эму / (I_вх * r_вх)",
        "T_эму": "L1 / r1",
        "T_вх": "L_вх / r_вх",
        "W_эму(s)": "K_эму / ((T_эму * s + 1)(T_вх * s + 1))",
        "W_двигатель(s)": "1.24 / (0.0076*s^2 + 0.4*s + 1)",
        "W_разомкнутая(s)": "W_эму(s) * W_двигатель(s) * K_ос",
        "W_замкнутая(s)": "W_разомкнутая(s) / (1 + W_разомкнутая(s))",
        "Hurwitz_D": "Определитель 3x3 по первым 4 коэффициентам"
    }
}

with open("результаты.json", "w", encoding="utf-8") as f:
    json.dump(результаты, f, ensure_ascii=False, indent=2)

print("Данные успешно сохранены в результаты.json (без переходной характеристики)")
