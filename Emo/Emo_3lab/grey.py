import struct

# Функция для преобразования числа с плавающей запятой в бинарный код
def float_to_bin(number):
    # Используем struct для преобразования float в 32-битный формат IEEE 754
    [d] = struct.unpack(">I", struct.pack(">f", number))
    return f'{d:032b}'

# Функция для кодирования бинарного числа в код Грея
def binary_to_gray(binary):
    binary = int(binary, 2)
    gray = binary ^ (binary >> 1)
    return f'{gray:032b}'

# Функция для декодирования кода Грея обратно в бинарный код
def gray_to_binary(gray):
    gray = int(gray, 2)
    binary = gray
    while gray > 0:
        gray >>= 1
        binary ^= gray
    return f'{binary:032b}'

# Функция для преобразования бинарного числа обратно в float
def bin_to_float(binary):
    bf = int(binary, 2).to_bytes(4, byteorder="big")
    return struct.unpack('>f', bf)[0]

# Пример использования
number = -12.34
print(f"Original number: {number}")

# Кодирование
binary = float_to_bin(number)
print(f"Binary: {binary}")

gray = binary_to_gray(binary)
print(f"Gray code: {gray}")

# Декодирование
decoded_binary = gray_to_binary(gray)
print(f"Decoded Binary: {decoded_binary}")

decoded_number = bin_to_float(decoded_binary)
print(f"Decoded Number: {decoded_number}")
