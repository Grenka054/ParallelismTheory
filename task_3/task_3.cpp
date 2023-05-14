#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <openacc.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifdef _FLOAT
    #define T float
    #define MAX std::fmaxf
    #define STOD std::stof
    #define cublascopy cublasScopy
    #define cublasaxpy cublasSaxpy
    #define cublasIamax cublasIsamax
#else
    #define T double
    #define MAX std::fmax
    #define STOD std::stod
    #define cublascopy cublasDcopy
    #define cublasaxpy cublasDaxpy
    #define cublasIamax cublasIdamax
#endif

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    {                                                                                              \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    }

// Вывести значения двумерного массива на gpu
void print_array_gpu(T *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            #pragma acc kernels present(A)
            printf("%.2f\t", A[IDX2C(i, j, size)]);
        }
        printf("\n");
    }
    printf("\n");
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
        A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend;                 // horizontal
        A[IDX2C(size - 1, i, size)] = A[IDX2C(size - 1, 0, size)] + addend;   // horizontal
        A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend;                 // vertical
        A[IDX2C(i, size - 1, size)] = A[IDX2C(0, size - 1, size)] + addend;   // vertical
    }
}

void calculate(const int net_size = 128, const int iter_max = 1e6, const T accuracy = 1e-6, const bool res = false)
{
	const size_t vec_size = net_size * net_size;

	// Создание 2-х двумерных матриц, одна будет считаться на основе другой
  	T* A = new T[vec_size],
	*Anew = new T[vec_size],
    * Adif = new T[vec_size];

	// Инициализация матриц
	std::memset(A, 0, vec_size * sizeof(T));
	initialize_array(A, net_size);
    std::memcpy(Anew, A, vec_size * sizeof(T));
    // Скаляр для вычитания, cublas требует указатель, поэтому выделим и под нее память
    const T alpha = -1;

    // Инкремент для матриц, в этой задаче 1
    const int inc = 1;

    // Индекс максимального элемента
    int max_idx = 0;

	// Текущая ошибка
    T error = accuracy + 1;

    // Указатель для swap
    T *temp;

    // Указатель на структуру, содержащую контекст
    cublasHandle_t handle;

    // Инициализация контекста
    CUBLAS_CHECK(cublasCreate(&handle));

	// Счетчик итераций
    int iter = 0;

	// Скопировать данные на девайс
	#pragma acc enter data copyin(A[:vec_size], Anew[:vec_size]) create(Adif[:vec_size])
    
    // // Вывести матрицу до начала основного алгоритма
    // if (res)
    // {
    //     std::cout << "--Borders--" << std::endl;
    //     print_array_gpu(A, net_size);
    // }

    for (iter = 0; iter < iter_max && error > accuracy; iter += net_size)
    {
        // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
        // Сначала посчитать net_size раз матрицы, после один раз посчитать ошибку
        for (uint32_t k = 0; k < net_size; ++k)
        {
            #pragma acc kernels loop independent collapse(2) present(A, Anew) async
            for (uint32_t i = 1; i < net_size - 1; i++)
                for (uint32_t j = 1; j < net_size - 1; j++)
                    Anew[IDX2C(j, i, net_size)] = (A[IDX2C(j + 1, i, net_size)] + A[IDX2C(j - 1, i, net_size)]
                                                + A[IDX2C(j, i - 1, net_size)] + A[IDX2C(j, i + 1, net_size)]) * 0.25;
            
            // swap(A, Anew)
            temp = A; A = Anew; Anew = temp;
        }
        #pragma acc wait
        #pragma acc data present(A, Anew, Adif) async
        #pragma acc host_data use_device(A, Anew, Adif)
        {
            // Adif = Anew
            CUBLAS_CHECK(cublascopy(handle, vec_size, Anew, inc, Adif, inc));
            // Adif = -1 * A + Adif 
            CUBLAS_CHECK(cublasaxpy(handle, vec_size, &alpha, A, inc, Adif, inc));
            // Получить индекс максимального абсолютного значения в Adif
            CUBLAS_CHECK(cublasIamax(handle, vec_size, Adif, inc, &max_idx));
        }
        #pragma acc update host(Adif[max_idx - 1])
            error = fabs(Adif[max_idx - 1]); // Fortran moment
    }
    #pragma acc wait
    std::cout << "Iter: " << iter << " Error: " << error << std::endl;
	if (res)
		print_array_gpu(A, net_size);
        
	#pragma acc exit data delete(A[:vec_size], Anew[:vec_size], Adif[:vec_size])
    cublasDestroy(handle);

    delete[] A;
    delete[] Anew;
    delete[] Adif;
}

int main(int argc, char *argv[])
{
	// Начать отсчет времени работы
    auto begin_main = std::chrono::steady_clock::now();

	// Парсинг аргументов командной строки
    int net_size = 128, iter_max = (int)1e6;
    T accuracy = 1e-6;
    bool res = false;
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
                std::cout << "Wrong args!\n";
                return -1;
            }
        }
    }

	// Заупстить решение задачи
    calculate(net_size, iter_max, accuracy, res);

	// Посчитать время выполнения
    auto end_main = std::chrono::steady_clock::now();
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
    return 0;
}