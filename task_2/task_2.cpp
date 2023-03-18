#include <iostream>
#include <cmath>
#include <chrono>

#ifdef _FLOAT
#define T float
#define MAX std::fmaxf
#else
#define T double
#define MAX std::fmax
#endif

// Вывести значения двумерного массива
void print_array(T **A, int size)
{
#pragma acc update host(A[:size][:size])
    std::cout.precision(4);
    for (int i = 0; i < size; i += 1)
    {
        for (int j = 0; j < size; j += 1)
            std::cout << A[i][j] << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T **A, int size)
{
    // Заполнение углов матрицы значениями
    A[0][0] = 10.0;
    A[0][size - 1] = 20.0;
    A[size - 1][size - 1] = 30.0;
    A[size - 1][0] = 20.0;

    // Заполнение периметра матрицы
    T step = 10.0 / (size - 1);
    for (int i = 1; i < size - 1; i++)
    {
        T addend = step * i;
        A[0][i] = A[0][0] + addend;               // horizontal
        A[size - 1][i] = A[size - 1][0] + addend; // horizontal
        A[i][0] = A[0][0] + addend;               // vertical
        A[i][size - 1] = A[0][size - 1] + addend; // vertical
    }

    // Заполнение 20-ю, чтобы сократить количество операций в основном алгоритме
    for (int i = 1; i < size - 1; i++)
        for (int j = 1; j < size - 1; j++)
            A[i][j] = 20.0;
}

// Очистка динамической памяти, выделенной под двумерую матрицу
void delete_2d_array(T **A, int size)
{
    for (int i = 0; i < size; i++)
        delete[] A[i];
    delete[] A;
}

// Основной алгоритм
void calculate(int net_size = 12, int iter_max = 1e6, T accuracy = 1e-6, bool res = false)
{
    // Создание 2-х двумерных матриц, одна будет считаться на основе другой
    T **Anew = new T *[net_size],
      **A = new T *[net_size];
    for (int i = 0; i < net_size; i++)
    {
        A[i] = new T[net_size];
        Anew[i] = new T[net_size];
    }

    // Инициализация матриц
    initialize_array(A, net_size);
    initialize_array(Anew, net_size);

    // Больше 30-и ошибки быть не должно
    T error = 30;
    // Счетчик итераций
    int iter = 0;
    // Указатель для swap
    T **temp;
    // Флаг обновления ошибки на хосте для обработки условием цикла
    bool update_flag = true;
#pragma acc data copyin(A[:net_size][:net_size], Anew[:net_size][:net_size]) copy(error)
    {
        while (error > accuracy && ++iter < iter_max)
        {
            // Сокращение количества обращений к CPU. Чем больше сетка, тем реже стоит сверять значения.
            update_flag = !(iter % net_size);
// Напомнить о массивах
#pragma acc data present(A, Anew)
// Асинхронно
#pragma acc kernels async(1)
            {
                // Вернуть ошибку на девайс и занулить
                if (update_flag)
                {
#pragma acc update device(error)
                    error = 0;
                }
#pragma acc loop independent collapse(2) reduction(max \
                                                   : error)
                for (int j = 1; j < net_size - 1; j++)
                {
                    for (int i = 1; i < net_size - 1; i++)
                    {
                        Anew[i][j] = (A[i + 1][j] + A[i - 1][j] + A[i][j - 1] + A[i][j + 1]) * 0.25;
                        if (update_flag)
                            error = std::max(error, std::abs(Anew[j][i] - A[j][i]));
                    }
                }
            }

            // swap(A, Anew)
            temp = A, A = Anew, Anew = temp;

            // Синхронизировать и отправить хосту для проверки условия
            if (update_flag)
            {
#pragma acc wait(1)
#pragma acc update self(error)
            }
        }
        if (res)
            print_array(A, net_size);
    }
    std::cout << "iter=" << iter << ",\terror=" << error << std::endl;

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
        if (!str.compare("-a"))
        {
#ifdef _FLOAT
            accuracy = std::stof(argv[arg + 1]);
#else
            accuracy = std::stod(argv[arg + 1]);
#endif
            arg++;
        }
        else if (!str.compare("-i"))
        {
            iter_max = (int)std::stod(argv[arg + 1]); // 1e6
            arg++;
        }
        else if (!str.compare("-s"))
        {
            net_size = std::stoi(argv[arg + 1]);
            arg++;
        }
        else if (!str.compare("-res"))
        {
            res = true;
        }
    }
    calculate(net_size, iter_max, accuracy, res);
    auto end_main = std::chrono::steady_clock::now();
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
    return 0;
}
