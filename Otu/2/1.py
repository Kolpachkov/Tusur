import numpy as np
import matplotlib.pyplot as plt

den_mikh = W_замкнутая.den[0][0]  # коэффициенты характеристического полинома
omega_m = np.logspace(-2, 3, 2000)  # частоты от 0.01 до 1000

s = 1j * omega_m
H = np.polyval(den_mikh, s)  # значения полинома на jω

x = H.real
y = H.imag

plt.figure()
plt.plot(x, y, label="Годограф Михайлова")
plt.scatter(x[0], y[0], color='green', label='Начало')
plt.scatter(x[-1], y[-1], color='red', label='Конец')

plt.text(x[0], y[0], f"({x[0]:.2f}, {y[0]:.2f})", fontsize=8, ha='right', va='bottom', color='green')
plt.text(x[-1], y[-1], f"({x[-1]:.2f}, {y[-1]:.2f})", fontsize=8, ha='left', va='top', color='red')

plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Годограф Михайлова (декартовые координаты)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
