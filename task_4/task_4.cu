/*
    На хосте только один массив, потому что второй
    временный, и он уже есть на девайсе.

    net_size - 2 потоков, поскольку при расчете
    не используется периметр, и я решил не создавать
    потоков, которые ничего полезного не делают.
*/

#include <chrono>
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

#ifdef _FLOAT
#define T float
#define MAX std::fmaxf
#define STOD std::stof
#else
#define T double
#define MAX std::fmax
#define STOD std::stod
#endif

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

// Вывести значения двумерного массива
void print_array(T *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
            printf("%.2f\t", A[IDX2C(i, j, size)]);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Посчитать матрицу
__global__ void calculate_matrix(T *A, T *Anew, uint32_t size)
{
    uint32_t i = blockIdx.x + 1;
    uint32_t j = threadIdx.x + 1;

    // Выход за границы массива или периметр - ничего не делать
    if (i > size - 2 || j > size - 2)
        return;

    Anew[IDX2C(i, j, size)] = (A[IDX2C(i + 1, j, size)] + A[IDX2C(i - 1, j, size)] + A[IDX2C(i, j - 1, size)] + A[IDX2C(i, j + 1, size)]) * 0.25;
}

// O = |A-B|
__global__ void count_matrix_difference(T *matrixA, T *matrixB, T *outputMatrix, uint32_t size)
{
    uint32_t i = blockIdx.x + 1;
    uint32_t j = threadIdx.x + 1;

    // Выход за границы массива или периметр - ничего не делать
    if (i > size - 2 || j > size - 2)
        return;

    uint32_t idx = IDX2C(i, j, blockDim.x);
    outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T *A, int size)
{
    // Заполнение углов матрицы значениями
    A[IDX2C(0, 0, size)] = 10.0;
    A[IDX2C(0, size - 1, size)] = 20.0;
    A[IDX2C(size - 1, 0, size)] = 20.0;
    A[IDX2C(size - 1, size - 1, size)] = 30.0;

    // Заполнение периметра матрицы
    T step = 10.0 / (size - 1);

    for (int i = 1; i < size - 1; ++i)
    {
        T addend = step * i;
        A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend;               // horizontal
        A[IDX2C(size - 1, i, size)] = A[IDX2C(size - 1, 0, size)] + addend; // horizontal
        A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend;               // vertical
        A[IDX2C(i, size - 1, size)] = A[IDX2C(0, size - 1, size)] + addend; // vertical
    }
}

// Основной алгоритм
void calculate(int net_size = 128, int iter_max = 1e6, T accuracy = 1e-6, bool res = false)
{
    cudaSetDevice(3);
    // Размер вектора - размер сетки в квадрате
    int vec_size = net_size * net_size;

    // Создание матрицы на хосте и 2-х на девайсе
    T *A, *A_dev, *Anew_dev;
    A = new T[vec_size];
    cudaMalloc(&A_dev, sizeof(T) * vec_size);    // Матрица
    cudaMalloc(&Anew_dev, sizeof(T) * vec_size); // Еще одна матрица

    // Инициализация матриц
    initialize_array(A, net_size);

    // Скопировать заполненные массивы с хоста на девайс
    cudaMemcpy(A_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(Anew_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice);

    // Вывод
    if (res)
    {
        cudaMemcpy(A, A_dev, sizeof(T) * vec_size, cudaMemcpyDeviceToHost);
        print_array(A, net_size);
    }

    // Текущая ошибка и матрица ошибок
    T error = 0;
    T *error_dev, *A_err;

    // Выделение памяти на девайсе
    cudaMalloc(&error_dev, sizeof(T));        // Ошибка (переменная)
    cudaMalloc(&A_err, sizeof(T) * vec_size); // Матрица ошибок

    // Указатель для swap
    T *temp;

    // Временный буфер для редукции и его размер
    T *reduction_bufer = NULL;
    size_t reduction_bufer_size = 0;

    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);

    // Выделение памяти под буфер
    cudaMalloc(&reduction_bufer, reduction_bufer_size);

    // Флаг обновления ошибки на хосте для обработки условием цикла
    bool update_flag = true;

    // Счетчик итераций
    int iter;

    for (iter = 0; iter < iter_max; ++iter)
    {
        // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
        update_flag = !(iter % net_size);

        calculate_matrix<<<net_size - 2, net_size - 2>>>(A_dev, Anew_dev, net_size);

        // swap(A_dev, Anew_dev)
        temp = A_dev, A_dev = Anew_dev, Anew_dev = temp;

        // Проверить ошибку
        if (update_flag)
        {
            count_matrix_difference<<<net_size - 2, net_size - 2>>>(A_dev, Anew_dev, A_err, net_size);

            // Найти максимум и положить в error_dev - аналог reduction (max : error_dev) в OpenACC
            cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);

            // Копировать ошибку с девайса на хост
            cudaMemcpy(&error, error_dev, sizeof(T), cudaMemcpyDeviceToHost);

            // Если ошибка не превышает точность, прекратить выполнение цикла
            if (error <= accuracy)
                break;
        }
    }

    std::cout.precision(2);
    // Вывод
    if (res)
    {
        cudaMemcpy(A, A_dev, sizeof(T) * vec_size, cudaMemcpyDeviceToHost);
        print_array(A, net_size);
    }
    std::cout << "iter=" << iter << ",\terror=" << error << std::endl;

    // Освобождение памяти
    cudaFree(reduction_bufer);
    cudaFree(A_err);
    cudaFree(A_dev);
    cudaFree(Anew_dev);
    free(A);
}

int main(int argc, char *argv[])
{
    auto begin_main = std::chrono::steady_clock::now();
    int net_size = 128, iter_max = (int)1e6;
    T accuracy = 1e-6;
    bool res = false;

    // Парсер
    for (int arg = 1; arg < argc; arg++)
    {
        std::string str = argv[arg];
        if (!str.compare("-res"))
            res = true;
        else
        {
            arg++;
            if (!str.compare("-a"))
                accuracy = STOD(argv[arg]);
            else if (!str.compare("-i"))
                iter_max = (int)std::stod(argv[arg]);
            else if (!str.compare("-s"))
                net_size = std::stoi(argv[arg]);
            else
            {
                std::cout << "Wrong args!";
                return -1;
            }
        }
    }

    calculate(net_size, iter_max, accuracy, res);
    auto end_main = std::chrono::steady_clock::now();
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
    return 0;
}