#include <iostream>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cmath> 


int findMinAbsElement_InlineAsm(const std::vector<int>& arr) {
    if (arr.empty()) {
        throw std::invalid_argument("Input array cannot be empty.");
    }

    const int* p_arr = arr.data();
    size_t size = arr.size();
    int result;

    asm (
        // --- Инициализация ---
        // Загружаем первый элемент массива arr[0] в регистр eax (будущий результат)
        "movl (%[p_arr]), %%eax \n\t"
        "movl %%eax, %%ebx \n\t"
        "cdq \n\t"
        "xorl %%edx, %%ebx \n\t"
        "subl %%edx, %%ebx \n\t"

        "movq $1, %%rcx \n\t"
    "loop_start:"
        "cmpq %%rcx, %[size] \n\t"
        // Если i >= size, выходим из цикла
        "jle loop_end \n\t"

        "movl (%[p_arr], %%rcx, 4), %%edx \n\t"
        "movl %%edx, %%esi \n\t"
        "sarl $31, %%esi \n\t"
        "xorl %%esi, %%edx \n\t"
        "subl %%esi, %%edx \n\t"


        "cmpl %%edx, %%ebx \n\t"
        "jle not_smaller \n\t"
        "movl %%edx, %%ebx \n\t"
        "movl (%[p_arr], %%rcx, 4), %%eax \n\t"
    "not_smaller:"
        // Увеличиваем счетчик i
        "incq %%rcx \n\t"
        // Возвращаемся в начало цикла
        "jmp loop_start \n\t"
    "loop_end:"
        // Сохраняем итоговый результат из eax в C++ переменную
        "movl %%eax, %[result] \n\t"
        : [result] "=r" (result)
        : [p_arr] "r" (p_arr), [size] "r" (size)
        : "%rax", "%rbx", "%rcx", "%rdx", "%rsi", "memory"
    );
    return result;
}

// --- ЗАДАНИЕ 2: Вычисление выражения с помощью FPU ---
double calculate_fpu(double A, double B, double C, double D) {
    double result;
    asm (
        // Команда fldl: загружает 64-битное вещественное число (double) на вершину стека FPU.
        // Стек FPU работает по принципу LIFO (Last-In, First-Out).
        "fldl   %[B]      \n\t"   // Загрузить B. Стек: ST(0)=B
        "fldl   %[A]      \n\t"   // Загрузить A. Стек: ST(0)=A, ST(1)=B
        // Команда fmulp: умножает ST(0) на ST(1), сохраняет результат в ST(1) и выталкивает ST(0) со стека.
        "fmulp  %%st(1)   \n\t"   // Стек: ST(0) = A * B
        // Команда faddl: прибавляет к вершине стека ST(0) значение из памяти.
        "faddl  %[C]      \n\t"   // Стек: ST(0) = A*B + C
        "fldl   %[D]      \n\t"   // Загрузить D. Стек: ST(0)=D, ST(1)=A*B+C
        // Команда fdivl: делит вершину стека ST(0) на значение из памяти.
        "fdivl  %[B]      \n\t"   // Стек: ST(0) = D/B, ST(1)=A*B+C
        // Команда faddp: складывает ST(0) и ST(1), сохраняет результат в ST(1) и выталкивает ST(0).
        "faddp  %%st(1)   \n\t"   // Стек: ST(0) = A*B+C + D/B
        "faddl  %[A]      \n\t"   // Прибавить A. Стек: ST(0) = A*B+C+D/B + A
        : "=t" (result)
        : [A]"m"(A), [B]"m"(B), [C]"m"(C), [D]"m"(D)
    );
    return result;
}

// --- ЗАДАНИЕ 3: Вычисление выражения с помощью MMX ---
void calculate_Y_mmx(int16_t* Y, const int8_t* A, const int8_t* B, const int16_t* C) {
    asm (
        // Команда movd: загружает 32 бита (4 байта) из памяти в 64-битный MMX-регистр.
        "movd       (%[A]), %%mm0      \n\t"   // mm0 = [0,0,0,0, a3,a2,a1,a0]
        "movd       (%[B]), %%mm1      \n\t"   // mm1 = [0,0,0,0, b3,b2,b1,b0]
        "pxor       %%mm7, %%mm7      \n\t"   
        "pcmpgtb    %%mm0, %%mm7      \n\t"
        "punpcklbw  %%mm7, %%mm0      \n\t"   
        "pcmpgtb    %%mm1, %%mm7      \n\t"   
        "punpcklbw  %%mm7, %%mm1      \n\t"   
        "pmullw     %%mm0, %%mm0      \n\t"   // mm0 = A * A
        "psubw      %%mm1, %%mm0      \n\t"   // mm0 = (A*A) - B
        "paddw      (%[C]), %%mm0      \n\t"   // mm0 = (A*A - B) + C
       
        "movq       %%mm0, (%[Y])      \n\t"
       
        "emms                         \n\t"
        : 
        : [Y]"r"(Y), [A]"r"(A), [B]"r"(B), [C]"r"(C)
        : "mm0", "mm1", "mm7", "memory"
    );
}

// --- ЗАДАНИЕ 4: Вычисление выражения с помощью SSE ---
void calculate_Y_sse(float* Y, const float* A, const float* B, const float* C, const float* D) {
    asm (
        "movaps (%[A]), %%xmm0      \n\t"   // xmm0 = [A3, A2, A1, A0]
        "movaps (%[B]), %%xmm1      \n\t"   // xmm1 = [B3, B2, B1, B0]
        "movaps (%[C]), %%xmm2      \n\t"   // xmm2 = [C3, C2, C1, C0]
        "movaps (%[D]), %%xmm3      \n\t"   // xmm3 = [D3, D2, D1, D0]

        // --- Выполняем вычисление: Y = A*B + C + D/B + A ---


        "mulps  %%xmm1, %%xmm0      \n\t"   // xmm0 = A * B


        "addps  %%xmm2, %%xmm0      \n\t"   // xmm0 = A*B + C

        "divps  %%xmm1, %%xmm3      \n\t"   // xmm3 = D / B

        "addps  %%xmm3, %%xmm0      \n\t"   // xmm0 = (A*B + C) + (D/B)

        "addps  (%[A]), %%xmm0      \n\t"   // xmm0 = (A*B + C + D/B) + A

        "movaps %%xmm0, (%[Y])      \n\t"
        : 
        : [Y]"r"(Y), [A]"r"(A), [B]"r"(B), [C]"r"(C), [D]"r"(D)
        : "xmm0", "xmm1", "xmm2", "xmm3", "memory"
    );
}

int main() {
    // --- ЗАДАНИЕ 1 ---
    std::cout << "--- ЗАДАНИЕ 1: Найти минимальный по модулю элемент ---" << std::endl;
    std::vector<int> numbers = {10, -15, 7, -3, 25, -2};
    std::cout << "Массив: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    try {
        int result = findMinAbsElement_InlineAsm(numbers);
        std::cout << "Наименьший модуль у числа: " << result << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
    }
    std::cout << "\n" << std::endl;

    // --- ЗАДАНИЕ 2 ---
    std::cout << "--- ЗАДАНИЕ 2: Вычисление выражения с помощью FPU ---" << std::endl;
    double fpu_varA = 3.0;
    double fpu_varB = 5.0;
    double fpu_varC = 10.0;
    double fpu_varD = 20.0;
    double fpu_y_cpp = (fpu_varA * fpu_varB) + fpu_varC + (fpu_varD / fpu_varB) + fpu_varA;
    double fpu_y_asm = calculate_fpu(fpu_varA, fpu_varB, fpu_varC, fpu_varD);
    std::cout << "A=" << fpu_varA << ", B=" << fpu_varB << ", C=" << fpu_varC << ", D=" << fpu_varD << std::endl;
    std::cout << "Результат на C++: Y = " << fpu_y_cpp << std::endl;
    std::cout << "Результат из функции FPU: Y = " << fpu_y_asm << std::endl;
    std::cout << "\n" << std::endl;

    // --- ЗАДАНИЕ 3 ---
    std::cout << "--- ЗАДАНИЕ 3: Вычисление выражения с помощью MMX ---" << std::endl;
    const int mmx_N = 8;
    alignas(16) int8_t mmx_A[mmx_N] = {-110, -3, 4, 10, 5, 6, -7, 8};
    alignas(16) int8_t mmx_B[mmx_N] = {1, 50, -2, 20, 10, 15, -126, 25};
    alignas(16) int16_t mmx_C[mmx_N] = {10, 20, 30, -50, 100, 120, 130, 140};
    alignas(16) int16_t mmx_Y_mmx[mmx_N];
    int16_t mmx_Y_cpp[mmx_N];

    for (int i = 0; i < mmx_N; i += 4) {
        calculate_Y_mmx(&mmx_Y_mmx[i], &mmx_A[i], &mmx_B[i], &mmx_C[i]);
    }
    for (int i = 0; i < mmx_N; ++i) {
        mmx_Y_cpp[i] = (int16_t)mmx_A[i] * (int16_t)mmx_A[i] - (int16_t)mmx_B[i] + mmx_C[i];
    }
    std::cout << "Результаты для Y = A^2 - B + C" << std::endl;
    std::cout << "Инд |    A |    B |     C | MMX Y | C++ Y |" << std::endl;
    for (int i = 0; i < mmx_N; ++i) {
        printf("%3d | %4d | %4d | %5d | %5d | %5d |\n", i, mmx_A[i], mmx_B[i], mmx_C[i], mmx_Y_mmx[i], mmx_Y_cpp[i]);
    }
    std::cout << "\n" << std::endl;

    // --- ЗАДАНИЕ 4 ---
    std::cout << "--- ЗАДАНИЕ 4: Вычисление выражения с помощью SSE ---" << std::endl;
    const int sse_N = 8;
    alignas(16) float sse_A[sse_N] = {1.0f, 2.2f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    alignas(16) float sse_B[sse_N] = {5.0f, 4.0f, 3.0f, 2.4f, 1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) float sse_C[sse_N] = {10.0f, 20.0f, 30.0f, 40.8f, 50.0f, 60.0f, 70.0f, 80.0f};
    alignas(16) float sse_D[sse_N] = {50.0f, 40.9f, 30.0f, 20.0f, 10.0f, 12.0f, 14.9f, 16.0f};
    alignas(16) float sse_Y_sse[sse_N];
    float sse_Y_cpp[sse_N];

    for (int i = 0; i < sse_N; i += 4) {
        calculate_Y_sse(&sse_Y_sse[i], &sse_A[i], &sse_B[i], &sse_C[i], &sse_D[i]);
    }
    for (int i = 0; i < sse_N; ++i) {
        sse_Y_cpp[i] = sse_A[i] * sse_B[i] + sse_C[i] + sse_D[i] / sse_B[i] + sse_A[i];
    }
    std::cout << "Результаты для Y = A*B + C + D/B + A (SSE)" << std::endl;
    std::cout << "Инд | SSE Y      | C++ Y      |" << std::endl;
    for (int i = 0; i < sse_N; ++i) {
        printf("%3d | %10.2f | %10.2f |\n", i, sse_Y_sse[i], sse_Y_cpp[i]);
    }

    return 0;
}