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

#define MIN(a,b) (((a)<(b))?(a):(b))

void print_device_properties(void)
{
    cudaDeviceProp deviceProp;
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0))
    {
        printf("Warp size in threads is %d.\n", deviceProp.warpSize);
        printf("Maximum size of each dimension of a block is %d, %d, %d.\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid is %d, %d, %d.\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum resident threads per multiprocessor is %d.\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of resident blocks per multiprocessor is %d.\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("_____________________________________________________________________________________________\n");
    }
}

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
__global__ void calculate_matrix(T *Anew, T *A, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Выход за границы массива или периметр - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i == 0 || j == 0)
        return;

    Anew[IDX2C(i, j, size)] = (A[IDX2C(i + 1, j, size)] + A[IDX2C(i - 1, j, size)] + A[IDX2C(i, j - 1, size)] + A[IDX2C(i, j + 1, size)]) * 0.25;
}

// O = |A-B|
__global__ void count_matrix_difference(T *matrixA, T *matrixB, T *outputMatrix, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Выход за границы массива или периметр - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i == 0 || j == 0)
        return;

    uint32_t idx = IDX2C(i, j, size);
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

    // Матрица на хосте (нужна только для инициализации и вывода) [Pinned]
    T *A;
    cudaMallocHost(&A, sizeof(T) * vec_size);    

    // Инициализация матрицы
    initialize_array(A, net_size);

    // Создание 2-х матриц на девайсе
    T *A_dev, *Anew_dev;
    cudaMalloc(&A_dev, sizeof(T) * vec_size);    // Матрица
    cudaMalloc(&Anew_dev, sizeof(T) * vec_size); // Еще одна матрица

    // Скопировать матрицу с хоста на матрицы на девайсе
    cudaMemcpy(A_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(Anew_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice);

    // Вывод
    if (res)
    {
        cudaMemcpy(A, A_dev, sizeof(T) * vec_size, cudaMemcpyDeviceToHost);
        print_array(A, net_size);
    }

    // Текущая ошибка
    T *error, *error_dev;
    cudaMallocHost(&error, sizeof(T));
    cudaMalloc(&error_dev, sizeof(T));        // Ошибка (переменная)
    *error = 0;

    // Матрица ошибок
    T *A_err;
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

    int threads_in_block = MIN(net_size, 32); // Потоков в одном блоке (32 * 32 максимум)
    int block_in_grid = ceil(net_size / threads_in_block); // Блоков в сетке (size / 32 максимум)

    // Счетчик итераций
    int iter;
    
    for (iter = 0; iter < iter_max; ++iter)
    {
        // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
        update_flag = !(iter % net_size);

        calculate_matrix<<<dim3(block_in_grid,block_in_grid), dim3(threads_in_block,threads_in_block)>>>(Anew_dev, A_dev, net_size);

        // swap(A_dev, Anew_dev)
        temp = A_dev, A_dev = Anew_dev, Anew_dev = temp;

        // Проверить ошибку
        if (update_flag)
        {
            count_matrix_difference<<<dim3(block_in_grid,block_in_grid), dim3(threads_in_block,threads_in_block)>>>(A_dev, Anew_dev, A_err, net_size);

            // Найти максимум и положить в error_dev - аналог reduction (max : error_dev) в OpenACC
            cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);

            // Копировать ошибку с девайса на хост
            cudaMemcpy(error, error_dev, sizeof(T), cudaMemcpyDeviceToHost);

            // Если ошибка не превышает точность, прекратить выполнение цикла
            if (*error <= accuracy)
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
    std::cout << "iter=" << iter << ",\terror=" << *error << std::endl;

    // Освобождение памяти
    cudaFree(reduction_bufer);
    cudaFree(A_err);
    cudaFree(A_dev);
    cudaFree(Anew_dev);
    cudaFreeHost(A);
    cudaFreeHost(error);
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
