import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy import optimize

class ControlSystem:
    def __init__(self):
        # Параметры системы
        self.K_em = 11
        self.T_v = 0.2  # постоянная времени обмотки управления
        self.T_e = 0.5  # постоянная времени якорной цепи
        self.K = 0.1    # коэффициент нагрузки
        self.T_e_load = 0.18
        self.T_n = 0.1   # постоянная времени нагрузки
        
        # Создание передаточных функций
        self.create_transfer_functions()
        
    def create_transfer_functions(self):
        # Передаточная функция ЭМУ
        self.W_em = ctrl.tf([self.K_em], [self.T_v*self.T_e, self.T_v+self.T_e, 1])
        
        # Передаточная функция нагрузки
        self.W_load = ctrl.tf([self.K*self.T_e_load, self.K], [self.T_n, 1])
        
        # Общая передаточная функция разомкнутой системы
        self.W_open = self.W_em * self.W_load
        
        # Передаточная функция замкнутой системы
        self.W_closed = self.W_open / (1 + self.W_open)
    
    def get_crossover_frequency(self, sys):
        """Вычисление частоты среза для заданной системы"""
        mag, phase, omega = ctrl.bode_plot(sys, plot=False)
        crossover_idx = np.where(mag <= 1)[0]
        if len(crossover_idx) > 0:
            return omega[crossover_idx[0]]
        return None
    
    def plot_bode(self, sys, title='ЛАЧХ и ЛФЧХ системы'):
        """Построение диаграммы Боде для системы"""
        plt.figure(figsize=(12, 8))
        
        # ЛАЧХ
        plt.subplot(2, 1, 1)
        mag, phase, omega = ctrl.bode_plot(sys, plot=False)
        plt.semilogx(omega, 20 * np.log10(mag))
        plt.grid(True, which='both')
        plt.ylabel('Амплитуда [дБ]')
        plt.title(title)
        
        # Отметка частоты среза
        w_c = self.get_crossover_frequency(sys)
        if w_c is not None:
            idx = np.argmin(np.abs(omega - w_c))
            plt.scatter(w_c, 20 * np.log10(mag[idx]), color='r')
            plt.axvline(w_c, color='r', linestyle='--', alpha=0.5)
            plt.text(w_c, 20 * np.log10(mag[idx]), f' ω_c={w_c:.2f} рад/с', 
                    verticalalignment='bottom')
        
        # ЛФЧХ
        plt.subplot(2, 1, 2)
        plt.semilogx(omega, phase * 180 / np.pi)
        plt.grid(True, which='both')
        plt.xlabel('Частота [рад/с]')
        plt.ylabel('Фаза [градусы]')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_system(self, plot=True, title='Переходный процесс системы'):
        # Анализ переходного процесса
        t = np.linspace(0, 2, 1000)
        t, y = ctrl.step_response(self.W_closed, t)
        
        # Расчет показателей качества
        y_ss = y[-1]
        y_max = max(y)
        sigma = (y_max - y_ss) / y_ss * 100 if y_ss != 0 else 0
        
        # Более точный расчет времени переходного процесса
        settled = np.where(np.abs(y - y_ss) <= 0.02 * y_ss)[0]
        if len(settled) > 0:
            cross_idx = np.where(np.abs(y - y_ss) > 0.02 * y_ss)[0]
            if len(cross_idx) > 0 and cross_idx[-1] > settled[0]:
                t_p = t[cross_idx[-1]]
            else:
                t_p = t[settled[0]]
        else:
            t_p = t[-1]
        
        # Расчет частоты среза
        w_c = self.get_crossover_frequency(self.W_open)
        
        if plot:
            # График переходного процесса
            plt.figure(figsize=(10, 6))
            plt.plot(t, y)
            plt.axhline(y_ss * 1.02, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y_ss * 0.98, color='r', linestyle='--', alpha=0.5)
            plt.title(title)
            plt.xlabel('Время, с')
            plt.ylabel('Выходное напряжение Uэ')
            plt.grid(True)
            plt.show()
            
            # ЛАЧХ и ЛФЧХ
            self.plot_bode(self.W_open, title='ЛАЧХ и ЛФЧХ исходной системы')
        
        print(f"Частота среза системы: {w_c:.2f} рад/с")
        return sigma, t_p, t, y, w_c
    
    def design_compensator(self, sigma=5, t_p=0.15, plot=True):
        # Расчет частоты среза
        if sigma <= 5:
            w_c = 4 * np.pi / t_p
        else:
            w_c = (2 * np.pi / t_p) * ((sigma + 45) / 20 - 2)
        
        print(f"\nРасчетная частота среза: {w_c:.2f} рад/с")
        
        # Параметры технического оптимума
        a = 2  # для σ = 4.3%
        T2 = 1 / (2 * w_c)  # предполагаем ω1в = 2ωс
        K = 1 / (a * T2)
        
        # Желаемая передаточная функция
        W_desired = ctrl.tf([K], [T2, 1, 0])
        
        # Передаточная функция корректирующего звена
        W_comp = W_desired / self.W_open
        
        if plot:
            # Анализ скорректированной системы
            W_open_corr = self.W_open * W_comp
            W_closed_corr = W_open_corr / (1 + W_open_corr)
            
            t = np.linspace(0, 0.5, 1000)
            t, y = ctrl.step_response(W_closed_corr, t)
            
            # Расчет частоты среза для скорректированной системы
            w_c_corr = self.get_crossover_frequency(W_open_corr)
            print(f"Фактическая частота среза после коррекции: {w_c_corr:.2f} рад/с")
            
            # График переходного процесса
            plt.figure(figsize=(10, 6))
            plt.plot(t, y)
            plt.title('Переходный процесс после коррекции')
            plt.xlabel('Время, с')
            plt.ylabel('Выходное напряжение Uэ')
            plt.grid(True)
            plt.show()
            
            # ЛАЧХ и ЛФЧХ
            self.plot_bode(W_open_corr, title='ЛАЧХ и ЛФЧХ после коррекции')
            self.plot_bode(W_comp, title='ЛАЧХ корректирующего звена')
        
        return W_comp
    
    def optimize_system(self, W_comp=None, sigma_max=10):
        # Функция для создания PID-регулятора с ограничениями
        def create_pid(Kp, Ki, Kd, N=10):
            Kp = max(Kp, 0.01)
            Ki = max(Ki, 0.01)
            Kd = max(Kd, 0.001)
            return ctrl.tf([Kd*N, Kp*N + Ki, Ki*N], [1, N, 0])
        
        def analyze_pid(params):
            try:
                Kp, Ki, Kd = params
                W_pid = create_pid(Kp, Ki, Kd)
                
                if W_comp is not None:
                    W_open_new = self.W_open * W_comp * W_pid
                else:
                    W_open_new = self.W_open * W_pid
                    
                W_closed_new = W_open_new / (1 + W_open_new)
                
                t = np.linspace(0, 1.0, 2000)
                t, y = ctrl.step_response(W_closed_new, t)
                y = np.clip(y, 0, 2*np.max(y))
                
                y_ss = y[-1]
                if y_ss < 1e-6:
                    return 1e6, 100, t[-1], t, y, None
                    
                y_max = np.max(y)
                sigma = (y_max - y_ss) / y_ss * 100
                
                settled = np.where(np.abs(y - y_ss) <= 0.02 * y_ss)[0]
                if len(settled) > 0:
                    cross_idx = np.where(np.abs(y - y_ss) > 0.02 * y_ss)[0]
                    if len(cross_idx) > 0 and cross_idx[-1] > settled[0]:
                        t_p = t[cross_idx[-1]]
                    else:
                        t_p = t[settled[0]]
                else:
                    t_p = t[-1]
                
                # Расчет частоты среза
                w_c = self.get_crossover_frequency(W_open_new)
                
                overshoot_penalty = max(0, sigma - sigma_max) * 5
                settling_time_penalty = t_p * 1
                control_effort = np.sum(np.abs(np.diff(y))) / len(y)
                
                fitness = overshoot_penalty + settling_time_penalty + control_effort
                
                return fitness, sigma, t_p, t, y, w_c
                
            except Exception as e:
                print(f"Ошибка при анализе: {e}")
                return 1e6, 100, 10, np.linspace(0, 1, 100), np.zeros(100), None
        
        bounds = [(0.1, 3), (0.1, 15), (0.01, 1.5)]
        initial_guess = [(0.5, 0.3, 0.1), (1.0, 0.5, 0.2), (0.8, 0.4, 0.15)]
        
        best_result = None
        best_fitness = float('inf')
        
        for guess in initial_guess:
            result = optimize.minimize(
                lambda params: analyze_pid(params)[0],
                guess,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 30}
            )
            
            if result.fun < best_fitness:
                best_fitness = result.fun
                best_result = result
        
        if best_result is None:
            best_result = optimize.differential_evolution(
                lambda params: analyze_pid(params)[0],
                bounds,
                strategy='best1bin',
                maxiter=50,
                popsize=10,
                tol=0.001,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42
            )
        
        best_params = best_result.x
        fitness, sigma, t_p, t, y, w_c = analyze_pid(best_params)
        
        print("\nРезультаты оптимизации:")
        print(f"Лучшие параметры PID: Kp={best_params[0]:.4f}, Ki={best_params[1]:.4f}, Kd={best_params[2]:.4f}")
        print(f"Перерегулирование: {sigma:.2f}%")
        print(f"Время переходного процесса: {t_p:.4f} с")
        print(f"Частота среза после оптимизации: {w_c:.2f} рад/с")
        
        # График переходного процесса
        plt.figure(figsize=(12, 6))
        plt.plot(t, y)
        plt.axhline(y[-1] * 1.02, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y[-1] * 0.98, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Оптимизированная система (σ={sigma:.1f}%, t_p={t_p:.3f}с)')
        plt.xlabel('Время, с')
        plt.ylabel('Выходное напряжение Uэ')
        plt.grid(True)
        
        textstr = '\n'.join((
            f'Kp = {best_params[0]:.3f}',
            f'Ki = {best_params[1]:.3f}',
            f'Kd = {best_params[2]:.3f}',
            f'σ = {sigma:.1f}%',
            f't_p = {t_p:.3f}с',
            f'ω_c = {w_c:.1f} рад/с'))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.75, 0.95, textstr, transform=plt.gca().transAxes,
                      fontsize=10, verticalalignment='top', bbox=props)
        
        plt.show()
        
        # ЛАЧХ и ЛФЧХ после оптимизации
        W_pid = create_pid(*best_params)
        if W_comp is not None:
            W_open_opt = self.W_open * W_comp * W_pid
        else:
            W_open_opt = self.W_open * W_pid
        
        self.plot_bode(W_open_opt, title='ЛАЧХ и ЛФЧХ после оптимизации PID')
        
        return best_params

if __name__ == "__main__":
    system = ControlSystem()
    
    print("1. Анализ исходной системы:")
    sigma_orig, t_p_orig, _, _, w_c_orig = system.analyze_system(plot=True, title='Исходная система')
    print(f"Исходная система: σ = {sigma_orig:.2f}%, t_p = {t_p_orig:.4f} с, ω_c = {w_c_orig:.2f} рад/с")
    
    print("\n2. Синтез корректирующего звена для σ ≤ 5%, t_p ≤ 0.15 с:")
    W_comp = system.design_compensator(sigma=5, t_p=0.15, plot=True)
    print("Передаточная функция корректирующего звена:")
    print(W_comp)
    
    print("\nАнализ системы с компенсатором:")
    W_open_corr = system.W_open * W_comp
    W_closed_corr = W_open_corr / (1 + W_open_corr)
    system.W_closed = W_closed_corr
    sigma_comp, t_p_comp, _, _, w_c_comp = system.analyze_system(plot=True, title='Система с компенсатором')
    print(f"Система с компенсатором: σ = {sigma_comp:.2f}%, t_p = {t_p_comp:.4f} с, ω_c = {w_c_comp:.2f} рад/с")
    system.create_transfer_functions()
    
    print("\n3. Оптимизация системы для σ ≤ 10%:")
    optimal_params = system.optimize_system(W_comp=W_comp, sigma_max=10)