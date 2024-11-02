import struct
 
def float_to_bin(number):
    
    [d] = struct.unpack(">I", struct.pack(">f", number))
    return f'{d:032b}'


def bin_to_float(binary):
    bf = int(binary, 2).to_bytes(4, byteorder="big")
    return struct.unpack('>f', bf)[0]


def binary_to_gray_custom(binary_string):
    L = len(binary_string)  
    binary_value = int(binary_string, 2)
    gray_value = binary_value ^ (binary_value >> 1)
    return f'{gray_value:0{L}b}'

def gray_to_binary_custom(gray_string):
    L = len(gray_string)  
    gray_value = int(gray_string, 2)
    binary_value = gray_value
    shift = 1
    while gray_value >> shift:
        binary_value ^= gray_value >> shift
        shift += 1
    return f'{binary_value:0{L}b}'


number = -12.3432085920385893
print(f"Original number: {number}")


binary = float_to_bin(number)
print(f"Binary: {binary}")

gray = binary_to_gray_custom(binary)
print(f"Gray code: {gray}")


decoded_binary = gray_to_binary_custom(gray)
print(f"Decoded Binary: {decoded_binary}")

decoded_number = bin_to_float(decoded_binary)
print(f"Decoded Number: {decoded_number}")
