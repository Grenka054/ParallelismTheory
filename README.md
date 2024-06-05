# Основные задачи (по 100 баллов)
## 1. Знакомство с OpenACC
Заполнить на графическом процессоре массив типа `float/double` значениями синуса (один период на всю длину массива). Размер массива - $10^7$ элементов. Для заполненного массива на графическом процессоре посчитать сумму всех элементов массива. Сравнить со значением, вычисленном на центральном процессоре. Сравнить результат для
массивов `float` и `double`.

При сборке программы для исполнения на GPU использовать компилятор `pgcc/pgc++` с ключами `-acc -Minfo=accel`. Необходимо разобраться с выводом о распараллеливании кода.

Произвести профилирование программы установив переменную окружения `PGI_ACC_TIME=1`. Необходимо понять сколько времени тратится на вычисление и сколько времени на передачу данных. 

Результаты работы, какие знания о работе программы на GPU были получены представить в виде отчета.

Возможные дополнительные вопросы:
* как выполняется параллельное суммирование, 
* почему в программе два цикла, а ядер (по выводу профилировщика) три
* почему сумма всех элементов не равна нулю

## 2. Уравнение теплопроводности
Реализовать решение уравнение теплопроводности (пятиточечный шаблон) в двумерной  
области на  равномерных сетках ($128^2$, $256^2$, $512^2$, $1024^2$). Граничные условия – линейная интерполяция между углами области. Значение в углах – 10, 20, 30, 20. Ограничить точность – $10^{-6}$ и максимальное число итераций – $10^6$. 

Параметры (точность, размер сетки, количество итераций) должны задаваться через 
параметры командной строки.

Вывод программы - количество итераций и достигнутое значение ошибки.

Перенести программу на GPU используя директивы OpenACC. Сравнить скорость работы для разных размеров сеток на центральном и графическом процессоре. При замерах времени на центральном процессоре  привести данные по  использованию нескольких ядер CPU (`-ta=multicore`).

Произвести профилирование программы с использованием NsightSystems. Произвести оптимизацию кода. Вопросы для анализа: что ограничивает производительность? Как  можно исправить ситуацию? Делает ли программа что-то лишнее?

При профилировании ограничивать количество итерации десятком - сотней. В противном случае файл с результатами профилирования будет очень большим, затруднит работу с ним, но прибавит понимания того, как работает программа.

Результат работы программы, анализ исходного кода и результат оптимизации (с графиками "до" и "после")  и пояснениями представить в виде отчета.

## 3. Оптимизированные библиотеки
Реализовать решение уравнение теплопроводности (пятиточечный шаблон) в двумерной  
области на  равномерных сетках ($128^2$, $256^2$, $512^2$, $1024^2$). Граничные условия – линейная интерполяция между углами области. Значение в углах – 10, 20, 30, 20. Ограничить точность – $10^{-6}$ и максимальное число итераций – $10^6$. 

Параметры (точность, размер сетки, количество итераций) должны задаваться через параметры командной строки.

Вывод программы - количество итераций и достигнутое значение ошибки.

Перенести программу на GPU используя директивы OpenACC. Операцию редукции (вычисление максимального значения ошибки) на графическом процессоре реализовать через вызовы функций из библиотеки cuBLAS

Сравнить скорость работы для разных размеров сеток на центральном и графическом процессоре (текущая реализация и реализация без библиотеки cuBLAS). При замерах времени на центральном процессоре  привести данные по  использованию нескольких ядер CPU (`-ta=multicore`).

Произвести профилирование программы с использованием NsightSystems. 

При профилировании ограничивать количество итераций десятком-сотней. В противном случае файл с результатами профилирования будет очень большим, затруднит работу с ним, но прибавит понимания того, как работает программа.

Отчет о проделанной работе и результатах представить в виде отчета.

## 4. Уравнение теплопроводности на CUDA
Реализовать решение уравнение теплопроводности (пятиточечный шаблон) в двумерной  
области на  равномерных сетках ($128^2$, $256^2$, $512^2$, $1024^2$). Граничные условия – линейная интерполяция между углами области. Значение в углах – 10, 20, 30, 20. Ограничить точность – $10^{-6}$ и максимальное число итераций – $10^6$. 

Параметры (точность, размер сетки, количество итераций) должны задаваться через параметры командной строки.

Вывод программы - количество итераций и достигнутое значение ошибки.

Перенести программу на GPU используя CUDA. Операцию редукции (подсчет максимальной ошибки) реализовать с использованием библиотеки CUB. 
  
Сравнить скорость работы для разных размеров сеток на графическом процессоре с предыдущей реализацией на OpenACC. 
  
Предоставить отчет, описывающий реализацию кода, результаты профилирования и сравнения с предыдущими реализациями.

## 5. Уравнение теплопроводности на нескольких GPU
Программа должна быть реализована на CUDA. Распараллеливание на несколько GPU должно производиться с использованием MPI. Операцию редукции (подсчет максимальной ошибки) в рамках одного MPI процесса реализовать с использованием библиотеки CUB. Подсчет глобального значения ошибки, обмен граничными условиями реализовать в двух вариантах: с использованием MPI и NCCL+MPI.

Сравнить скорость работы и масштабирование для разных размеров сеток на разном  
количестве графических процессоров (1, 2, 4). Обратить внимание на корректность отображения MPI процессов на ядра центрального процессора и корректный выбор графических процессоров. Результаты, их анализ и выводы представить в виде отчета.

Доп. материалы
[MULTI-GPU PROGRAMMING MODELS](https://on-demand.gputechconf.com/gtc/2017/presentation/s7142-jiri-kraus-multi-gpu-programming-models.pdf)

## 6. Обучение нейронной сети
Написать программу на CUDA, реализующую простую нейронную сеть, заданную скриптом ниже
```
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
         super(Net, self).__init__()
         self.fc1 = nn.Linear(32**2, 16**2) # входной слой
         self.fc2 = nn.Linear(16**2, 4**2) # скрытый слой
         self.fc3 = nn.Linear(4 ** 2,1) # скрытый слой

# прямое распространение информации

    def forward(self, x):
         sigmoid = nn.Sigmoid()
         x = sigmoid(self.fc1(x))
         x = sigmoid(self.fc2(x))
         x = sigmoid(self.fc3(x))
         return x


input_layer = torch.rand(32**2) # входные данные нейронной сети
net = Net() # создание объекта "нейронная сеть"
result = net(input_layer) # запуск прямого распространения информации
print(result)
```
Создать бинарные файлы под входные значения, значения весовых коэффициентов. Сравнить корректность работы программы на CUDA и на Python. 

Пересчет значений со слоя на слой реализовать через вызовы из библиотек cuBLAS, cuDNN.

# Дополнительные задачи
## 1. Построение фрактала (30 баллов)
[Задание](https://drive.google.com/file/d/1mJ6JCO-DfGCZ4pbJf-WTgvBXfKUvGtOQ/view?usp=drive_link)

## 2. Варианты организации процессов (30 баллов)
Для получения дополнительных баллов за курс просьба в свободной форме порассуждать:
- об организации ручного производства пельменей
- об организации ручного производства керамической посуды
- о правилах поведения на эскалаторе с точки зрения минимизации задержек и/или пропускной споcобности

Какие формы параллелизма возможны? Какие "процессоры" с точки зрения классификации Флинна под это подходят? Как для подобных систем работает закон Амдала?

## 3. Conway's Game of Life (дополнительно) (40 баллов)
Описание задачи в презентации.
[Game of Life](https://docs.google.com/presentation/d/1HCUTZC9YLnM3jJFOTJuz-bafjQ43edVg/edit?usp=drive_link&ouid=113729624199134893752&rtpof=true&sd=true)
Указать время работы, для сетки 8192x8192 (20 итераций).
В командной строке должны задаваться размер сетки и количество итераций.
Начальную расстановку клеток задать алгоритмом (с привязкой к координатами), либо рандомно.
