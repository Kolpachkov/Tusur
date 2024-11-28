import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Определяем функцию
def f(x1, x2):
    return 2*x1**2 + x1*x2 + x2**2 - 3*x1

# Создаем массивы значений x1 и x2
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
x1, x2 = np.meshgrid(x1, x2)

# Вычисляем значения функции
z = f(x1, x2)

# Создаем 3D-график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis')

# Отмечаем оптимальную точку
x_opt, y_opt = 0.857, -0.429
z_opt = f(x_opt, y_opt)
ax.scatter(x_opt, y_opt, z_opt, color='r', s=50)

ax.set_title("График функции f(x) = 2x1^2 + x1x2 + x2^2 - 3x1")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x)")

plt.show()
