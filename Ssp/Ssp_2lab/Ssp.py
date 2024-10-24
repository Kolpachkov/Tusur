import threading
import numpy as np
import time
import matplotlib.pyplot as plt

first = True
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

    column_sum = 0
    processed_columns = np.zeros(matrix.shape[1], dtype=bool)  

    row_chunks = np.array_split(matrix, num_threads, axis=0)
    col_indices = np.arange(matrix.shape[1])  
    
    threads = []
    for submatrix in row_chunks:
        t = threading.Thread(target=process_submatrix, args=(submatrix, col_indices))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def sequential_processing(matrix):
    cols_with_zeros = np.any(matrix == 0, axis=0).nonzero()[0]
    return sum(cols_with_zeros)

def benchmark(matrix_size, num_threads):
    matrix = np.random.randint(0, 127, size=(matrix_size, matrix_size), dtype=np.byte)
    
    # Последовательная обработка
    start_time = time.time()
    sequential_result = sequential_processing(matrix)
    sequential_time = time.time() - start_time

    # Параллельная обработка
    start_time = time.time()
    parallel_processing(matrix, num_threads)
    parallel_time = time.time() - start_time

    
    assert column_sum == sequential_result, f"Результаты не совпадают! {column_sum} != {sequential_result}"
    if parallel_time == 0 or sequential_time == 0: 
        parallel_time = 0.0001
        sequential_time = 0.0001

    parallel_speed = (matrix_size * matrix_size) / parallel_time
    sequential_speed = (matrix_size * matrix_size) / sequential_time

    return sequential_speed, parallel_speed


matrix_sizes = [ 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,20000,30000]
sequential_speeds = []
parallel_speeds = []

for size in matrix_sizes:
    seq_speed, par_speed = benchmark(size, 4)
    sequential_speeds.append(seq_speed)
    parallel_speeds.append(par_speed)
    
    if seq_speed < par_speed and first:
        plt.scatter(size, par_speed, color='red', label=f"Размер матрицы, где параллельное быстрее ({size})")
        first = False

    print(f"Матрица {size}x{size}: Скорость последовательной обработки: {seq_speed:.2f}, Скорость параллельной обработки: {par_speed:.2f}")

plt.plot(matrix_sizes, sequential_speeds, label='Последовательная обработка')
plt.plot(matrix_sizes, parallel_speeds, label='Параллельная обработка (4 потока)')
plt.xlabel('Размер матрицы')
plt.ylabel('Скорость (элементы/секунды)')
plt.title('Сравнение скорости обработки матриц в зависимости от их размера')
plt.legend()
plt.grid(True)
plt.show()
