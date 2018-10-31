# coding=utf-8
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

N = 1000  # 进化代数
size = 10  # 种群数量
train = 9  # 被接续列车数
ntrain = 6  # 需要调整的列车数
pc = 0.8  # 交叉概率
pm = 0.2  # 变异概率
ha = 4
hd = 4
tmin = 14
T = 120
arrival = [20, 24, 37, 45, 51, 63, 78, 82, 90]
running = [150, 130, 145, 140, 150, 133]
population = np.zeros([size, ntrain])
new_population = np.zeros([size, ntrain])
# 最优解的值
result_num = np.zeros(N)
# 最优的数组
opt_arr = np.zeros(ntrain)
# 最优解
opt = 10000
# 约束条件的数组
x = np.zeros((size, ntrain, train))
y = np.zeros((size, ntrain, train))

f = np.zeros(size)
p = np.zeros(size)


# 随机生成整数
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


# 初始化种群
def init_population():
    # 全局变量
    global f, population, new_population, x, y
    # 初始化
    while True:
        for i in range(0, size):
            for j in range(0, ntrain):
                if j == 0:
                    population[i][j] = random_int_list(1, T / ntrain * (j + 1), 1)[0]
                else:
                    population[i][j] = random_int_list(population[i][j - 1] + hd, T / ntrain * (j + 1), 1)[0]
        # print(population)

        # 判断列车是否接续成功
        y = np.zeros((size, ntrain, train))
        x = np.zeros((size, ntrain, train))
        for i in range(0, size):
            for j in range(0, ntrain):
                for k in range(0, train):
                    if population[i][j] - arrival[k] >= tmin:
                        y[i][j][k] = 1
                    else:
                        y[i][j][k] = 0

        for i in range(0, size):
            for j in range(0, ntrain):
                for k in range(0, train):
                    if j == 0:
                        x[i][j][k] = y[i][j][k]
                    else:
                        x[i][j][k] = y[i][j][k] - y[i][j - 1][k]

        # 计算适应度函数值
        f = np.ones(size)
        # 不满足约束条件的个体，将其适应度函数值设为0
        for i in range(0, size):
            for k in range(0, train):
                sum_col = 0
                for j in range(0, ntrain):
                    sum_col += x[i][j][k]
                if sum_col == 0:
                    f[i] = 0
                    break
            for j in range(1, ntrain):
                if population[i][j] - population[i][j - 1] < hd:
                    f[i] = 0
                    break
            for j in range(0, ntrain):
                if population[i][j] <= 0:
                    f[i] = 0
                    break
            for j in range(1, ntrain):
                if population[i][j] + running[j] - (population[i][j - 1] + running[j - 1]) < hd:
                    f[i] = 0
                    break
            if f[i] != 0:
                f[i] = 0
                for j in range(0, ntrain):
                    for k in range(0, train):
                        f[i] += x[i][j][k] * (running[j] + population[i][j] - arrival[k])

        init = False
        for i in f:
            if i != 0:
                init = True
        if init:
            break


# 利用轮盘赌法进行选择操作
def choose():
    global new_population, population, p
    s = 0
    while True:
        if s == size:
            break
        num = random.random()
        if num < p[0]:
            for k in range(0, ntrain):
                new_population[s][k] = population[0][k]
            s += 1
        else:
            for i in range(0, size - 1, 1):
                if p[i] <= num < p[i + 1]:
                    for k in range(0, ntrain):
                        new_population[s][k] = population[i + 1][k]
                    s += 1
                    break


def cross():
    global new_population, population
    # 交叉
    exchange = []
    for i in range(0, size):
        exchange.append(i)
    random.shuffle(exchange)
    for i in range(0, int(size * pc / 2)):
        local = random_int_list(1, ntrain - 1, 1)[0]
        for j in range(local, len(new_population[0])):
            mid = new_population[exchange[i * 2]][j]
            new_population[exchange[i * 2]][j] = new_population[exchange[i * 2 + 1]][j]
            new_population[exchange[i * 2 + 1]][j] = mid


def variation():
    global new_population, population
    exchange = []
    for i in range(0, size):
        exchange.append(i)
    random.shuffle(exchange)
    for i in range(0, int(size * pm)):
        random_local = random_int_list(0, ntrain - 1, 1)[0]
        new_population[exchange[i]][random_local] = new_population[exchange[i]][random_local] + \
                                                    random_int_list(-5, 5, 1)[0]


def exchange_value():
    global new_population, population
    # 交换赋值，进入下一轮循环
    for i in range(0, size):
        for j in range(0, ntrain):
            population[i][j] = new_population[i][j]
            new_population[i][j] = 0


# 计算适应度函数
def calculation_fitness():
    # 全局变量
    global f, population, new_population, x, y
    # 计算适应度函数
    y = np.zeros((size, ntrain, train))
    x = np.zeros((size, ntrain, train))
    for i in range(0, size):
        for j in range(0, ntrain):
            for k in range(0, train):
                if population[i][j] - arrival[k] >= tmin:
                    y[i][j][k] = 1
                else:
                    y[i][j][k] = 0

    for i in range(0, size):
        for j in range(0, ntrain):
            for k in range(0, train):
                if j == 0:
                    x[i][j][k] = y[i][j][k]
                else:
                    x[i][j][k] = y[i][j][k] - y[i][j - 1][k]

    f = np.ones(size)
    for i in range(0, size):
        for k in range(0, train):
            sum_col = 0
            for j in range(0, ntrain):
                sum_col += x[i][j][k]
            if sum_col == 0:  # 不满足约束条件的个体，将其适应度函数值设为0
                f[i] = 0
                break
        for j in range(1, ntrain):
            if population[i][j] - population[i][j - 1] < hd:
                f[i] = 0
                break
        for j in range(0, ntrain):
            if population[i][j] <= 0:
                f[i] = 0
                break
        for j in range(1, ntrain):
            if population[i][j] + running[j] - (population[i][j - 1] + running[j - 1]) < hd:
                f[i] = 0
                break
        if f[i] != 0:
            f[i] = 0
            for j in range(0, ntrain):
                for k in range(0, train):
                    f[i] += x[i][j][k] * (running[j] + population[i][j] - arrival[k])


def save_best():
    global population, new_population, opt_arr, f, opt
    all_zero = False
    all_zero_index = 0
    for i in range(0, size):
        if f[i] == 0:
            all_zero = True
            all_zero_index = i

    if all_zero:
        for i in range(0, ntrain):
            population[all_zero_index][i] = opt_arr[i]
    else:
        max_next_p = f[0]
        max_next_l = 0
        for i in range(0, size):
            if f[i] > max_next_p:
                max_next_p = f[i]
                max_next_l = i

        if 1 / opt > max_next_p:
            for i in range(0, ntrain):
                population[max_next_l][i] = opt_arr[i]


# 初始化选择数组
def init_p():
    global f, p
    # 选择操作
    for i in range(0, size):
        if f[i] != 0:
            f[i] = 1 / f[i]
    F = 0
    for i in range(0, size):
        F += f[i]
    p = np.zeros(size)

    for i in range(0, size):
        if i == 0:
            p[i] = f[i] / F
        else:
            p[i] = f[i] / F + p[i - 1]


# 选择最优值并刷新
def select_fresh(n):
    global opt, population, new_population, f, opt_arr, result_num
    # 选择最优解
    min_aim = opt
    min_aim_loc = 0
    for i in range(0, size):
        if f[i] != 0:
            if 1 / f[i] < min_aim:
                min_aim = 1 / f[i]
                min_aim_loc = i

    # print("min_aim")
    # print(min_aim)

    # 刷新最优值
    if n == 0:
        opt = min_aim
        for i in range(0, ntrain):
            opt_arr[i] = population[min_aim_loc][i]
    elif opt > min_aim:
        opt = min_aim
        # 将最优的一代保留下来
        for i in range(0, ntrain):
            opt_arr[i] = population[min_aim_loc][i]

    print("opt")
    print(opt)

    # 给result_num
    result_num[n] = opt


# 核心算法
def function_ga(N):
    for n in range(0, N):
        # 选择最优值并刷新
        select_fresh(n)
        print("第" + str(n) + "代的" + "最优解")
        print(opt_arr)
        # 轮盘赌
        choose()
        # 交叉
        cross()
        # 变异
        variation()
        # 交换赋值
        exchange_value()
        # 计算适应度f
        calculation_fitness()
        # 保存最优解
        save_best()
        # 计算适应度函数
        calculation_fitness()
        # 计算轮盘赌选择数组p
        init_p()


# 主函数
if __name__ == '__main__':
    # 初始化
    init_population()
    # 计算轮盘赌选择数组p
    init_p()
    # 遗传算法核心函数
    function_ga(N)

    # 画图
    x = range(N)
    y = copy.deepcopy(result_num)  # X轴，Y轴数据
    plt.figure(figsize=(10, 6))  # 创建绘图对象
    plt.plot(x, y, "*-b", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色点线，线宽度）
    # plt.ylim(0, 200) #Y轴坐标轴范围
    plt.xlabel("generation")  # X轴标签
    plt.ylabel("Price")  # Y轴标签
    plt.title("GA")  # 图标题
    plt.show()  # 显示图
    # plt.savefig("line.jpg") #保存图
