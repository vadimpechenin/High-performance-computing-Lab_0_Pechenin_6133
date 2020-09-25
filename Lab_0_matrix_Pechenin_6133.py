"""
Лабораторная работа № 1 по курсу "Облачные и высокопроизводительные вычисления"
Реализовано 2 функции перемножения матриц: на CPU и на GPU с применением CUDA
CPU: Intel(R) Core(TM) i5-3470 @3.2 GHz, 4 ядра
GPU: NVIDIA GeForce GTX 1050 Ti (CUDA 10.1)

@author: Vadim Pechenin, group 6133
"""
#1. Подключение библеотек
import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
# Библиотека для работы с эксель
from openpyxl import Workbook
from datetime import datetime

def matrix_creation_cpu(index):
    # Создание матрицы с помощью numpy
    matrix_a = np.ones((index, index), 'f')
    matrix_b = np.ones((index, index), 'f')
    return matrix_a, matrix_b

def matrix_creation_gpu(index):
    # Создание матрицы с помощью cupy
    matrix_a = cp.ones((index, index), 'f')
    matrix_b = cp.ones((index, index), 'f')
    return matrix_a, matrix_b

def dot_computation_cpu(matrix_a,matrix_b):
    # Умножение матриц с помощью numpy
    s = time.time()
    dot_product_cpu = np.dot(matrix_a, matrix_b)
    e = time.time()
    time_of_dot = e - s
    return dot_product_cpu, time_of_dot

def dot_computation_gpu(matrix_a,matrix_b):
    # Умножение матриц с помощью cupy
    s = time.time()
    dot_computation_gpu = cp.dot(matrix_a, matrix_b)
    e = time.time()
    time_of_dot = e - s
    return dot_computation_gpu, time_of_dot

class ParallelDot():
    """Класс для выполнения лабораторной работы"""
    def __init__(self,size,table_of_results,path_file):
        self.size = size
        self.table_of_results = table_of_results
        self.path_file = path_file
    def computations_of_experiment(self):
        #Общие вычисления, формирование таблицы результатов
        for index in range(len(self.size)):
            self.table_of_results[0, index]=self.size[index]
            matrix_a, matrix_b = matrix_creation_cpu(self.size[index])
            dot_product_cpu, self.table_of_results[1, index] = dot_computation_cpu(matrix_a,matrix_b)
            matrix_a, matrix_b = matrix_creation_gpu(self.size[index])
            dot_product_gpu, self.table_of_results[2, index] = dot_computation_gpu(matrix_a,matrix_b)
            # Проверка точности
            check_matrix = np.ones((self.size[index], self.size[index]), 'f') * self.size[index]
            deviation_matrix = dot_product_cpu - check_matrix
            self.table_of_results[3, index] = np.absolute(deviation_matrix).argmax()
            check_matrixc = cp.ones((self.size[index], self.size[index]), 'f') * self.size[index]
            deviation_matrix = dot_product_gpu - check_matrixc
            self.table_of_results[4, index] = np.absolute(deviation_matrix).argmax()
            # Ускорение
            if ( self.table_of_results[1, index] > 0):
                self.table_of_results[5, index] = ((self.table_of_results[1, index] - self.table_of_results[2, index])/
                                                   self.table_of_results[1, index]*100)
            else:
                self.table_of_results[5, index] = ((self.table_of_results[1, index] - self.table_of_results[2, index])/
                                                   0.00001 * 100)

    def save_computations(self):
        # Создать рабочую книгу в Excel:
        dt = datetime.now()
        wb = Workbook()
        sheet = wb.active
        sheet.title = self.path_file

        # Добавить заголовки в рабочую книгу Excel:
        k=0
        sheet['A' + str(1)] = 'Размер матрицы'
        sheet['B' + str(1)] = 'Время CPU, с'
        sheet['C' + str(1)] = 'Время GPU, с'
        sheet['D' + str(1)] = 'Погрешность CPU'
        sheet['E' + str(1)] = 'Погрешность GPU'
        sheet['F' + str(1)] = 'Ускорение GPU относительно CPU'
        # Заполнить данными
        for item in self.table_of_results:
            for j in range(len(item)):
                if (k==0):
                    sheet['A' + str(j+2)] = item[j]
                elif (k==1):
                    sheet['B' + str(j + 2)] = item[j]
                elif (k==2):
                    sheet['C' + str(j + 2)] = item[j]
                elif (k==3):
                    sheet['D' + str(j + 2)] = item[j]
                elif (k==4):
                    sheet['E' + str(j + 2)] = item[j]
                else:
                    sheet['F' + str(j + 2)] = item[j]

            k=k+1
        # Сохранить файл:
        filename = self.path_file + '_' + dt.strftime("%Y%m%d_%I%M%S") + '.xlsx'
        wb.save(filename)


    def plot_results(self):
        # 3. Визуализация
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(u'Результаты по производительности')
        ax.plot(self.table_of_results[0, :], self.table_of_results[1, :], 'ko-', color='grey', label=r'$CPU$')
        ax.plot(self.table_of_results[0, :], self.table_of_results[2, :], 'gs-', label=r'$GPU$')
        ax.legend(loc='best')
        plt.xlabel("Размер матриц", fontsize=14, fontstyle="normal")
        plt.ylabel("Время расчета, с", fontsize=14, fontstyle="normal")
        fig.show()
        fig.savefig('Визуализация результатов.jpeg', dpi=300)


# Создание входного значения размеров
size = []
[size.append(idx) for idx in range(100,2001,100)]
table_of_results = np.zeros((6,len(size)))
path_file = 'Table_of_results'
parallel_dot = ParallelDot(size,table_of_results,path_file)
parallel_dot.computations_of_experiment()
parallel_dot.save_computations()
parallel_dot.plot_results()