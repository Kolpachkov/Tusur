import threading
import numpy as np
import time
import matplotlib.pyplot as plt

first =True
column_sum = 0

result_lock = threading.Lock()

processed_columns = None

def process_submatrix(submatrix, col_indices):
    global column_sum
    global processed_columns

    
    cols_with_zeros = np.any(submatrix == 0, axis=0).nonzero()[0]
    
    
    with result_lock:
        for col in cols_with_zeros:
            if not processed_columns[col]:  
                column_sum += col_indices[col]
                processed_columns[col] = True  

def parallel_processing(matrix, num_threads):
    global column_sum
    global processed_columns

    column_sum = 0  # Сбрасываем результат
    processed_columns = np.zeros(matrix.shape[1], dtype=bool)  # Инициализируем массив для отслеживания

    # Определим подматрицы для каждого потока (по строкам)
    row_chunks = np.array_split(matrix, num_threads, axis=0)
    col_indices = np.arange(matrix.shape[1])  # Индексы всех столбцов
    
    threads = []
    for submatrix in row_chunks:
        t = threading.Thread(target=process_submatrix, args=(submatrix, col_indices))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def sequential_processing(matrix):
    # Поиск столбцов с хотя бы одним нулем
    cols_with_zeros = np.any(matrix == 0, axis=0).nonzero()[0]
    
    return sum(cols_with_zeros)

def benchmark(matrix_size, num_threads):
    # Создаем случайную матрицу целочисленного типа (однобайтные значения)
    matrix = np.random.randint(0, 127, size=(matrix_size, matrix_size), dtype=np.byte)
    
    
    # Последовательная обработка
    start_time = time.time()
    sequential_result = sequential_processing(matrix)
    sequential_time = time.time() - start_time

    # Параллельная обработка
    start_time = time.time()
    parallel_processing(matrix, num_threads)
    parallel_time = time.time() - start_time

    # Проверка на совпадение результатов
    assert column_sum == sequential_result, f"Результаты не совпадают! {column_sum} != {sequential_result}"

    return sequential_time, parallel_time

# Построение графика
matrix_sizes = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
sequential_times = []
parallel_times = []

for size in matrix_sizes:
    seq_time, par_time = benchmark(size, 4)
    sequential_times.append(seq_time)
    parallel_times.append(par_time)
    if (seq_time > par_time and first ==True):
        plt.scatter(size, par_time, color='red', label=f"Размер матрицы,где параллельное быстрее({size})")
        first =False
    print(f"Матрица {size}x{size}: Последовательное время: {seq_time:.4f}, Параллельное время: {par_time:.4f}")

plt.plot(matrix_sizes, sequential_times, label='Последовательная обработка')
plt.plot(matrix_sizes, parallel_times, label='Параллельная обработка (4 потока)')
plt.xlabel('Размер матрицы')
plt.ylabel('Время (секунды)')
plt.title('Зависимость времени обработки матрицы от её размера')
plt.legend()
plt.grid(True)
plt.show()
import threading
import numpy as np
import time
import matplotlib.pyplot as plt

first =True
column_sum = 0

result_lock = threading.Lock()

processed_columns = None

def process_submatrix(submatrix, col_indices):
    global column_sum
    global processed_columns

    
    cols_with_zeros = np.any(submatrix == 0, axis=0).nonzero()[0]
    
    
    with result_lock:
        for col in cols_with_zeros:
            if not processed_columns[col]:  
                column_sum += col_indices[col]
                processed_columns[col] = True  

def parallel_processing(matrix, num_threads):
    global column_sum
    global processed_columns

    column_sum = 0  # Сбрасываем результат
    processed_columns = np.zeros(matrix.shape[1], dtype=bool)  # Инициализируем массив для отслеживания

    # Определим подматрицы для каждого потока (по строкам)
    row_chunks = np.array_split(matrix, num_threads, axis=0)
    col_indices = np.arange(matrix.shape[1])  # Индексы всех столбцов
    
    threads = []
    for submatrix in row_chunks:
        t = threading.Thread(target=process_submatrix, args=(submatrix, col_indices))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def sequential_processing(matrix):
    # Поиск столбцов с хотя бы одним нулем
    cols_with_zeros = np.any(matrix == 0, axis=0).nonzero()[0]
    
    return sum(cols_with_zeros)

def benchmark(matrix_size, num_threads):
    # Создаем случайную матрицу целочисленного типа (однобайтные значения)
    matrix = np.random.randint(0, 127, size=(matrix_size, matrix_size), dtype=np.byte)
    
    
    # Последовательная обработка
    start_time = time.time()
    sequential_result = sequential_processing(matrix)
    sequential_time = time.time() - start_time

    # Параллельная обработка
    start_time = time.time()
    parallel_processing(matrix, num_threads)
    parallel_time = time.time() - start_time

    # Проверка на совпадение результатов
    assert column_sum == sequential_result, f"Результаты не совпадают! {column_sum} != {sequential_result}"

    return sequential_time, parallel_time

# Построение графика
matrix_sizes = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
sequential_times = []
parallel_times = []

for size in matrix_sizes:
    seq_time, par_time = benchmark(size, 4)
    sequential_times.append(seq_time)
    parallel_times.append(par_time)
    if (seq_time > par_time and first ==True):
        plt.scatter(size, par_time, color='red', label=f"Размер матрицы,где параллельное быстрее({size})")
        first =False
    print(f"Матрица {size}x{size}: Последовательное время: {seq_time:.4f}, Параллельное время: {par_time:.4f}")

plt.plot(matrix_sizes, sequential_times, label='Последовательная обработка')
plt.plot(matrix_sizes, parallel_times, label='Параллельная обработка (4 потока)')
plt.xlabel('Размер матрицы')
plt.ylabel('Время (секунды)')
plt.title('Зависимость времени обработки матрицы от её размера')
plt.legend()
plt.grid(True)
plt.show()
