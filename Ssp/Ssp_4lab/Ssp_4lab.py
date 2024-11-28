import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def f(x):
    return [
        np.sin(x[0] + 1) - x[1] - 1.2,
        2*x[0] + np.cos(x[1]) - 2
    ]

x0 = [0, 0]

solution = fsolve(f, x0)

print("Решение системы:")
print(solution)

x = np.linspace(-20, 20, 400)
y = np.linspace(-20, 20, 400)


y1 = np.sin(x + 1) - 1.2

x2 = (2 - np.cos(y)) / 2


plt.figure(figsize=(10, 5))


plt.plot(x, y1, label="y = sin(x + 1) - 1.2")

plt.plot(x2, y, label="x = (2 - cos(y)) / 2")

plt.plot(solution[0], solution[1], 'ro') 
plt.text(solution[0], solution[1], f'({solution[0]:.2f}, {solution[1]:.2f})', fontsize=12, color='red')


plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Комбинация графиков: y от x и x от y с точкой пересечения")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.show()
