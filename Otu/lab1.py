import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Чтение данных из CSV
data = pd.read_csv('C:/Users/multi/Desktop/Tusur/Otu/data.csv', delimiter=',')

# Коэффициенты передаточной функции
for i in range(8):
    name = data.iloc[i, 1]
    b0 = data.iloc[i, 5]
    b1 = data.iloc[i, 6]
    a0 = data.iloc[i, 2]
    a1 = data.iloc[i, 3]
    a2 = data.iloc[i, 4]
    print(f"\nАнализ системы: {name}\n")

    # Если название звена "Идеальное дифференцирующее" или "Идеальное интегрирующее"
    if name == 'Идеальное дифференцирующее':
        num = [0, 0, 1]  # Числитель: 1
        den = [1e-6, 3]  # Знаменатель: 1e-6*s + 1 (маленькое значение)
    else:
        num = [b0, b1]      # соответствует 0*s + 1
        den = [a0, a1, a2]  # соответствует 1*s**2 + 2*s + 1

    system = ctrl.TransferFunction(num, den)
    # Получение частотной характеристики
    omega_vals = np.logspace(-4, 2, 100)
    mag, phase, omega_response = ctrl.frequency_response(system, omega_vals)

    # Преобразование частот из радиан в герцы
    omega_response_hz = omega_response / (2 * np.pi)

    # 1. Частота среза (по амплитудной характеристике на 0 дБ)
    cutoff_idx = np.where(20 * np.log10(mag) <= 0)[0]
    if len(cutoff_idx) > 0:
        cutoff_freq = omega_response_hz[cutoff_idx[0]]
        print(f"Частота среза для системы '{name}': {cutoff_freq} Гц")
    else:
        print(f"Частота среза для системы '{name}' не найдена в заданном диапазоне.")

    
    mag_diff = np.diff(np.log10(mag))  
    threshold = 0.03  
    conjugate_idx = np.where(np.abs(mag_diff) > threshold)[0]
    
    if len(conjugate_idx) > 0:
        conjugate_freq = omega_response_hz[conjugate_idx[0]]
        print(f"Частота сопряжения для системы '{name}': {conjugate_freq} Гц")
    else:
        print(f"Частота сопряжения для системы '{name}' не найдена в заданном диапазоне.")


    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    fig.suptitle(f'Анализ системы: {name}', fontsize=16)

    axs[0, 0].semilogx(omega_response_hz, 20 * np.log10(mag))
    axs[0, 0].set_title('Боде диаграмма - Амплитудная характеристика')
    axs[0, 0].set_xlabel('Частота [Гц]')
    axs[0, 0].set_ylabel('Амплитуда [дБ]')

    if len(cutoff_idx) > 0:
        axs[0, 0].axvline(x=cutoff_freq, color='r', linestyle='--', label=f'Частота среза: {cutoff_freq:.2f} Гц')
    
    if len(conjugate_idx) > 0:
        axs[0, 0].axvline(x=conjugate_freq, color='g', linestyle='--', label=f'Частота сопряжения: {conjugate_freq:.2f} Гц')

    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].semilogx(omega_response_hz, phase)
    axs[0, 1].set_title('Боде диаграмма - Фазовая характеристика')
    axs[0, 1].set_xlabel('Частота [Гц]')
    axs[0, 1].set_ylabel('Фаза [градус]')
    axs[0, 1].grid(True)

    axs[1, 0].plot(mag * np.cos(phase), mag * np.sin(phase), label='Нюйквистова диаграмма')
    axs[1, 0].set_title('Нюйквистова диаграмма')
    axs[1, 0].set_xlabel('Действительная часть')
    axs[1, 0].set_ylabel('Мнимая часть')
    axs[1, 0].grid(True)

   
    t, y = ctrl.step_response(system)
    axs[1, 1].plot(t, y)
    axs[1, 1].set_title('Переходный процесс (шага)')
    axs[1, 1].set_xlabel('Время [с]')
    axs[1, 1].set_ylabel('Отклик')
    axs[1, 1].grid(True)

    plt.tight_layout()

   
    plt.savefig(f'C:/Users/multi/Desktop/Tusur/Otu/response_plots_{name}.png')
    plt.close(fig)  
    print(f'Графики для системы "{name}" сохранены.')
