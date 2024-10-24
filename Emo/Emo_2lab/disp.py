import csv
import math

def calculate_variance_and_std_dev(csv_file, column_name):
    best_f_values = []

    
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        
        
        for row in reader:
            best_f_values.append(float(row[column_name]))

    
    if len(best_f_values) > 1:
        n = len(best_f_values)
        mean = sum(best_f_values) / n  
        variance = sum((x - mean) ** 2 for x in best_f_values) / (n - 1)  # Выборочная дисперсия
        std_dev = math.sqrt(variance)  # Стандартное отклонение
        return variance, std_dev
    else:
        return None, None

csv_file = 'genetic_algorithm_bin_results.csv'
column_name = 'Best f(x)'

variance, std_dev = calculate_variance_and_std_dev(csv_file, column_name)

if variance is not None:
    print(f'Выборочная дисперсия (Бинарное) "{column_name}": {variance}')
    print(f'Стандартное отклонение (Бинарное) "{column_name}": {std_dev}')

