import networkx as nx
import numpy as np
import pulp  # 导入 pulp库
from matplotlib import pyplot as plt


def sample():
    lbp = pulp.LpProblem("lbp", sense=pulp.LpMinimize)  # 定义问题 1，求最大值

    L = pulp.LpVariable("L", lowBound=0, cat="Integer")
    # 设置目标函数 f(x)
    lbp += (L)

    # 定义变量xij, 表示machine_i到job_j有一条边
    # 循环遍历machine和job，生成变量
    for i in range(machine_num):
        for j in range(job_num):
            exec('x{}{} = pulp.LpVariable("x{}{}", lowBound=0, cat="Integer")'.format(i + 1, j + 1, i + 1, j + 1))

    # weight是长度为job_num的随机数组，范围为3-10
    weight = np.random.randint(3, 10, job_num)
    # print weight
    print("weight: ", weight)

    machine_assign_job = [[] for _ in range(machine_num)]
    for i in range(machine_num):
        # randonm choose 1-job_num number from [1,2, ..., job_num]
        machine_assign_job[i] = np.random.choice(range(1, job_num + 1), np.random.randint(1, job_num + 1),
                                                 replace=False)
        machine_assign_job[i].sort()
        print("machine_assign_job[{}]: ".format(i + 1), machine_assign_job[i])

    # 等式约束, 即job的权重和等于weight
    print('equal constraint')
    for i in range(job_num):
        cmd = 'lbp += ('
        ok = False
        for j in range(len(machine_assign_job)):
            if i + 1 in machine_assign_job[j]:
                ok = True
                cmd += f'x{j + 1}{i + 1} + '
        # job没有分配给任何machine，直接跳过，该样本不合法
        if not ok:
            continue
        cmd = cmd[:-3]
        cmd += f' == {weight[i]})'
        exec(cmd)
        print(cmd)

    # 不等式约束，即machine的权重和小于等于L
    print('not equal constraint')
    for i in range(machine_num):
        cmd = 'lbp += ('
        for j in range(len(machine_assign_job[i])):
            cmd += f'x{i + 1}{machine_assign_job[i][j]} + '
        cmd = cmd[:-3]
        cmd += f' <= {L})'
        exec(cmd)
        print(cmd)

    lbp.solve()
    print(lbp.name)  # 输出求解状态
    print("Status:", pulp.LpStatus[lbp.status])  # 输出求解状态
    for v in lbp.variables():
        print(v.name, "=", int(v.varValue))  # 输出每个变量的最优值
    print("F1(x)=", pulp.value(lbp.objective))  # 输出最优解的目标函数值

    G = nx.Graph()
    plt.rcParams['figure.figsize'] = (24, 24)

    pos = {}
    for i in range(job_num):
        G.add_node(i + 1, weight=weight[i])
        pos[i + 1] = (1, job_num - i)

    # machine_cost为长度为machine_num的全0数组
    machine_cost = np.zeros(machine_num, dtype=np.int32)
    for v in lbp.variables():
        if len(v.name) > 1 and int(v.varValue) >= 1:
            job_id = int(v.name[2])
            machine_id = int(v.name[1])
            machine_cost[machine_id - 1] += int(v.varValue)

    for i in range(machine_num):
        G.add_node(i + 1 + job_num, weight=machine_cost[i])
        pos[i + 1 + job_num] = (2, machine_num - i + 1)

    # 最后的结果
    G.add_node(job_num + machine_num + 1, weight=int(L.varValue))
    pos[job_num + machine_num + 1] = (3, 3.5)

    # 取前6个节点加入到job_nodes中
    job_nodes = list(G.nodes)[:job_num]
    # 取后4个节点加入到machine_nodes中
    machine_nodes = list(G.nodes)[job_num:-1]
    final_node = list(G.nodes)[-1]

    for v in lbp.variables():
        if len(v.name) > 1 and int(v.varValue) >= 1:
            print(v.name, "=", int(v.varValue))  # 输出每个变量的最优值
            job_id = int(v.name[2])
            machine_id = int(v.name[1])
            G.add_edge(job_num + machine_id, job_id, weight=int(v.varValue))

    # 显示job
    nx.draw_networkx_nodes(G, pos, nodelist=job_nodes, node_size=2000, node_shape='o', node_color='r', label=None)
    # node显示权重而不是显示名称
    nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['weight'] for node in job_nodes}, font_size=32)

    #  显示machine
    nx.draw_networkx_nodes(G, pos, nodelist=machine_nodes, node_size=2000, node_shape='s', node_color='g')
    # node显示权重而不是显示名称
    nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['weight'] for node in machine_nodes}, font_size=32)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=2)
    #  显示edge的权重
    nx.draw_networkx_edge_labels(G, pos, label_pos=0.9,
                                 edge_labels={(u, v): G.edges[u, v]['weight'] for u, v in G.edges},
                                 font_size=16, verticalalignment='bottom', horizontalalignment='right', alpha=1)

    # labels
    matrix = nx.to_numpy_matrix(G)
    print(matrix)

    plt.axis('off')
    plt.savefig("net.jpg")
    plt.show()

    # 判断无向图matrix中是否存在交叉环
    def has_cycle(matrix):
        for i in range(job_num):
            for j in range(machine_num + job_num):
                if matrix[i, j] >= 1:

                    for p in range(job_num):
                        if p == i:
                            continue

                        for q in range(machine_num + job_num):
                            if q == j:
                                continue

                            if matrix[i, q] >= 1 and matrix[p, j] >= 1 and matrix[p, q] >= 1:
                                print(f'({i + 1}, {j + 1}) and ({p + 1}, {q + 1})')
                                return True

        return False

    is_cycle = has_cycle(matrix)
    return is_cycle


# main函数
if __name__ == '__main__':
    job_num = 6
    machine_num = 4
    while True:
        is_cycle = sample()
        if is_cycle:
            break
