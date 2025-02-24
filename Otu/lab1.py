import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Чтение данных из CSV
data = pd.read_csv('C:/Users/multi/Tusur/Otu/data.csv', delimiter=',')

# Коэффициенты передаточной функции
for i in range(8):
    name = data.iloc[i, 1]
    b0 = data.iloc[i, 5]
    b1 = data.iloc[i, 6]
    a0 = data.iloc[i, 2]
    a1 = data.iloc[i, 3]
    a2 = data.iloc[i, 4]
    print(name, a0, a1, a2, b0, b1)

    # Если название звена "Идеальное дифференцирующее" или "Идеальное интегрирующее"
    if name == 'Идеальное дифференцирующее':
        # Добавляем маленькое значение в знаменатель
        num = [0, 0, 1]  # Числитель: 1
        den = [1e-6, 3]  # Знаменатель: 1e-6*s + 1 (маленькое значение)
    else:
        # Обычные системы
        num = [b0, b1]      # соответствует 0*s + 1
        den = [a0, a1, a2]  # соответствует 1*s**2 + 2*s + 1

    system = ctrl.TransferFunction(num, den)

    # Настройка фигуры с несколькими подграфиками
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Устанавливаем заголовок для всей фигуры
    fig.suptitle(f'Анализ системы: {name}', fontsize=16)

    # Построение Боде-диаграммы вручную
    mag, phase, omega = ctrl.bode(system, dB=True, omega_limits=(0.1, 100), plot=False)

    # Амплитудная характеристика
    axs[0, 0].semilogx(omega, 20 * np.log10(mag))
    axs[0, 0].set_title('Боде диаграмма - Амплитудная характеристика')
    axs[0, 0].set_xlabel('Частота [рад/с]')
    axs[0, 0].set_ylabel('Амплитуда [дБ]')
    axs[0, 0].grid(True)

    # Фазовая характеристика
    axs[0, 1].semilogx(omega, phase)
    axs[0, 1].set_title('Боде диаграмма - Фазовая характеристика')
    axs[0, 1].set_xlabel('Частота [рад/с]')
    axs[0, 1].set_ylabel('Фаза [градус]')
    axs[0, 1].grid(True)

    # Построение Нюйквистовой диаграммы вручную
    omega_vals = np.logspace(-4, 2, 100)

    # Получение частотной характеристики
    mag, phase, omega_response = ctrl.frequency_response(system, omega_vals)

    # Строим Нюйквистову диаграмму
    axs[1, 0].plot(mag * np.cos(phase), mag * np.sin(phase), label='Нюйквистова диаграмма')
    axs[1, 0].set_title('Нюйквистова диаграмма')
    axs[1, 0].set_xlabel('Действительная часть')
    axs[1, 0].set_ylabel('Мнимая часть')
    axs[1, 0].grid(True)

    # Построение переходного процесса
    t, y = ctrl.step_response(system)
    axs[1, 1].plot(t, y)
    axs[1, 1].set_title('Переходный процесс (шага)')
    axs[1, 1].set_xlabel('Время [с]')
    axs[1, 1].set_ylabel('Отклик')
    axs[1, 1].grid(True)

    # Отображение всех графиков
    plt.tight_layout()

    # Сохранение изображения в файл PNG с названием системы
    plt.savefig(f'C:/Users/multi/Tusur/Otu/response_plots_{name}.png')
    plt.close(fig)  # Закрытие фигуры после сохранения, чтобы избежать перекрытия
    print(f'Графики для системы "{name}" сохранены.')
