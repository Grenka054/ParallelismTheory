#include <chrono>
#include <cmath>
#include <iostream>

#ifdef _FLOAT
    #define T float
    #define MAX std::fmaxf
    #define STOD std::stof
#else
    #define T double
    #define MAX std::fmax
    #define STOD std::stod
#endif

// Вывести значения двумерного массива
void print_array(T **A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            // Значение с GPU
            #pragma acc kernels present(A)
            printf("%.2f\t", A[i][j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T **A, int size)
{
    // Инициализируется матрица на GPU
    #pragma acc parallel present(A)
    {
        // Заполнение углов матрицы значениями
        A[0][0] = 10.0;
        A[0][size - 1] = 20.0;
        A[size - 1][size - 1] = 30.0;
        A[size - 1][0] = 20.0;

        // Заполнение периметра матрицы
        T step = 10.0 / (size - 1);
        #pragma acc loop independent
        for (int i = 1; i < size - 1; ++i)
        {
            T addend = step * i;
            A[0][i] = A[0][0] + addend;               // horizontal
            A[size - 1][i] = A[size - 1][0] + addend; // horizontal
            A[i][0] = A[0][0] + addend;               // vertical
            A[i][size - 1] = A[0][size - 1] + addend; // vertical
        }

        // Заполнение 20-ю, чтобы сократить количество операций в основном алгоритме
        #pragma acc loop independent collapse(2)
        for (int i = 1; i < size - 1; ++i)
            for (int j = 1; j < size - 1; ++j)
                A[i][j] = 20.0;
    }
}

// Очистка динамической памяти, выделенной под двумерую матрицу
void delete_2d_array(T **A, int size)
{
    for (int i = 0; i < size; i++)
        delete[] A[i];
    delete[] A;
}

// Основной алгоритм
void calculate(int net_size = 128, int iter_max = 1e6, T accuracy = 1e-6, bool res = false)
{
    // Создание 2-х двумерных матриц, одна будет считаться на основе другой
    T **Anew = new T *[net_size],
      **A = new T *[net_size];
    for (int i = 0; i < net_size; i++)
    {
        A[i] = new T[net_size];
        Anew[i] = new T[net_size];
    }

    #pragma acc enter data create(A[:net_size][:net_size], Anew[:net_size][:net_size])

    // Инициализация матриц
    initialize_array(A, net_size);
    initialize_array(Anew, net_size);
    
    // Текущая ошибка
    T error = 0;
    // Счетчик итераций
    int iter;
    // Указатель для swap
    T **temp;
    // Флаг обновления ошибки на хосте для обработки условием цикла
    bool update_flag = true;

    #pragma acc data copy(error)
    {
        for (iter = 0; iter < iter_max; ++iter)
        {
            // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
            update_flag = !(iter % net_size);

            if (update_flag)
            {
                // зануление ошибки на GPU
                #pragma acc kernels
                error = 0;
            }

            // Распараллелить циклы с редукцией и запустить асинхронные ядра
            #pragma acc kernels loop independent collapse(2) reduction(max : error) present(A, Anew) async(1)
            for (int i = 1; i < net_size - 1; i++)
                for (int j = 1; j < net_size - 1; j++)
                    Anew[i][j] = (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]) * 0.25;

            // swap(A, Anew)
            temp = A, A = Anew, Anew = temp;
            // Проверить ошибку
            if (update_flag)
            {
                for (int i = 1; i < net_size - 1; i++)
                    for (int j = 1; j < net_size - 1; j++)
                        // Пересчитать ошибку
                        error = std::max(error, std::abs(Anew[i][j] - A[i][j]));
                // Синхронизация и обновление ошибки на хосте
                #pragma acc update host(error) wait(1)
                // Если ошибка не превышает точность, прекратить выполнение цикла
                if (error <= accuracy)
                    break;
            }
        }
        // Синхронизация
        #pragma acc wait(1)
    }
    std::cout.precision(2);
    if (res)
        print_array(A, net_size);
    std::cout << "iter=" << iter << ",\terror=" << error << std::endl;

    #pragma acc exit data delete (A[:net_size][:net_size], Anew[:net_size][:net_size])
    delete_2d_array(A, net_size);
    delete_2d_array(Anew, net_size);
}

int main(int argc, char *argv[])
{
    auto begin_main = std::chrono::steady_clock::now();
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
