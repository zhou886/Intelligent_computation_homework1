import numpy as np
from network import Network
from train import train


def cross_operation(parent1: list, parent2: list):
    child1 = list(parent1)
    child2 = list(parent2)

    c = 0; p = 0; f = 0
    for i in range(len(child1)):
        if child1[i][0] == 'C':
            while (c < len(parent2)):
                if parent2[c][0] == 'C':
                    child1[i] = list(parent2[c])
                    c += 1
                    break
                c += 1
        elif child1[i][0] == 'P':
            while (p < len(parent2)):
                if parent2[p][0] == 'P':
                    child1[i] = list(parent2[p])
                    p += 1
                    break
                p += 1
        else:
            while (f < len(parent2)):
                if parent2[f][0] == 'F':
                    child1[i] = list(parent2[f])
                    f += 1
                    break
                f += 1
    
    c = 0; p = 0; f = 0
    for i in range(len(child2)):
        if child2[i][0] == 'C':
            while (c < len(parent1)):
                if parent1[c][0] == 'C':
                    child2[i] = list(parent1[c])
                    c += 1
                    break
                c += 1
        elif child2[i][0] == 'P':
            while (p < len(parent1)):
                if parent1[p][0] == 'P':
                    child2[i] = list(parent1[p])
                    p += 1
                    break
                p += 1
        else:
            while (f < len(parent1)):
                if parent1[f][0] == 'F':
                    child2[i] = list(parent1[f])
                    f += 1
                    break
                f += 1
    
    return child1, child2

def main():
    '''
    遗传算法的主要部分
    '''

    # 生成种群
    population = []

    # 超参数设置
    Pc = 0.5    # 交叉概率
    Pm = 0.05   # 变异概率
    population_size = 10    # 种群大小
    iteration_size = 100    # 遗传迭代次数

    # 优化结果
    result = [-1000000, []]

    # 随机生成种群
    for i in range(population_size):
        individual = []

        # 生成卷积层和池化层
        N = np.random.randint(1, 7)
        for j in range(N):
            rand = np.random.rand()
            tmp = []
            if rand > 0.5:
                # 卷积层
                tmp.append('C')

                kernerl_size = round(np.random.normal(3, 1))
                if kernerl_size <= 0:
                    kernerl_size = 1
                if kernerl_size % 2 == 0:
                    kernerl_size += 1
                tmp.append(kernerl_size)

                out_channels = round(np.random.normal(32, 8))
                if out_channels <= 0:
                    out_channels = 1
                tmp.append(out_channels)
            else:
                # 池化层
                tmp.append('P')

                kernerl_size = round(np.random.normal(2, 1))
                if kernerl_size <= 0:
                    kernerl_size = 1
                tmp.append(kernerl_size)

            individual.append(tmp)

        # 生成全连接层
        M = np.random.randint(1, 4)
        for j in range(M):
            tmp = ['F']
            if j < M-1:
                nerual_size = round(np.random.normal(128, 64))
                if nerual_size <= 0:
                    nerual_size = 1
                tmp.append(nerual_size)
            else:
                tmp.append(10)
            individual.append(tmp)
        population.append(individual)

    for ind in population:
        print(ind)

    # 进行迭代
    for iteration in range(iteration_size):
        print(iteration)
        fitness = []
        for i in range(population_size):
            individual_fitness_list = []

            # 对每个个体分别构造出对应的网络模型并进行5次训练
            for cnt in range(5):
                network_module = Network(population[i])
                loss = train(network_module)
                individual_fitness_list.append(loss)

            # 取5次训练结果LOSS的中位数作为该个体的适应度
            individual_fitness_list.sort()
            fitness.append(individual_fitness_list[2])

            # 和结果比较并更新结果
            if individual_fitness_list[2] > result[0]:
                result[0] = individual_fitness_list[2]
                result[1] = list(population[i])
            

        total_fitness = np.sum(fitness)
        counter = 0
        population_selected_probability = [0]
        next_general_population = []
        for i in range(population_size):
            counter += 1.0*fitness[i]/total_fitness
            population_selected_probability.append(counter)

        # 选择算子
        for i in range(population_size):
            tmp_probability = np.random.rand()
            for j in range(population_size):
                if tmp_probability > population_selected_probability[j] and tmp_probability < population_selected_probability[j+1]:
                    next_general_population.append(list(population[j]))
                    break

        # 交叉算子
        parent_id_list = []
        for i in range(population_size):
            tmp_probability = np.random.rand()
            if tmp_probability < Pc:
                parent_id_list.append(i)
            if len(parent_id_list) == 2:
                child1, child2 = cross_operation(next_general_population[parent_id_list[0]], next_general_population[parent_id_list[1]])
                next_general_population[parent_id_list[0]] = list(child1)
                next_general_population[parent_id_list[1]] = list(child2)
                parent_id_list.clear()
        
        # 变异算子 保留

        population = list(next_general_population)



if __name__ == '__main__':
    main()
