import networkx as nx
import numpy as np
import pulp  # 导入 pulp库
from matplotlib import pyplot as plt

job_num = 6
machine_num = 4

lbp = pulp.LpProblem("lbp", sense=pulp.LpMinimize)  # 定义问题 1，求最大值
# 定义变量 4个机器， 6个job。 xij 表示第i个机器上第j个job的时间
x11 = pulp.LpVariable("x11", lowBound=0, cat="Integer")
x12 = pulp.LpVariable("x12", lowBound=0, cat="Integer")
x13 = pulp.LpVariable("x13", lowBound=0, cat="Integer")
x14 = pulp.LpVariable("x14", lowBound=0, cat="Integer")
x15 = pulp.LpVariable("x15", lowBound=0, cat="Integer")
x16 = pulp.LpVariable("x16", lowBound=0, cat="Integer")

# 生成x21到x26
x21 = pulp.LpVariable("x21", lowBound=0, cat="Integer")
x22 = pulp.LpVariable("x22", lowBound=0, cat="Integer")
x23 = pulp.LpVariable("x23", lowBound=0, cat="Integer")
x24 = pulp.LpVariable("x24", lowBound=0, cat="Integer")
x25 = pulp.LpVariable("x25", lowBound=0, cat="Integer")
x26 = pulp.LpVariable("x26", lowBound=0, cat="Integer")

# 生成x31到x36
x31 = pulp.LpVariable("x31", lowBound=0, cat="Integer")
x32 = pulp.LpVariable("x32", lowBound=0, cat="Integer")
x33 = pulp.LpVariable("x33", lowBound=0, cat="Integer")
x34 = pulp.LpVariable("x34", lowBound=0, cat="Integer")
x35 = pulp.LpVariable("x35", lowBound=0, cat="Integer")
x36 = pulp.LpVariable("x36", lowBound=0, cat="Integer")

# 生成x41到x46
x41 = pulp.LpVariable("x41", lowBound=0, cat="Integer")
x42 = pulp.LpVariable("x42", lowBound=0, cat="Integer")
x43 = pulp.LpVariable("x43", lowBound=0, cat="Integer")
x44 = pulp.LpVariable("x44", lowBound=0, cat="Integer")
x45 = pulp.LpVariable("x45", lowBound=0, cat="Integer")
x46 = pulp.LpVariable("x46", lowBound=0, cat="Integer")

L = pulp.LpVariable("L", lowBound=0, cat="Integer")

# 设置目标函数 f(x)
lbp += (L)
# # 等式约束
# exec('lbp += (x11 == 3)')
# exec('lbp += (x12 + x22 == 8)')
# exec('lbp += (x13 + x23 + x33 == 5)')
# exec('lbp += (x34 + x44 == 7)')
# exec('lbp += (x35 + x45 == 6)')
# exec('lbp += (x46 == 3)')
#
# # 不等式约束，可以用exec生成代码
# exec('lbp += (x11 + x12 + x13 <= L)')
# exec('lbp += (x22 + x23 <= L)')
# exec('lbp += (x33 + x34 + x35 <= L)')
# exec('lbp += (x44 + x45 + x46 <= L)')

# weight是长度为4的随机数组，范围为1-10
weight = np.random.randint(1, 10, 6)
print('Weight:', weight)

print('job_assign_machine')
job_assign_machine = [[] for _ in range(job_num)]
for i in range(job_num):
    # random choose 1-4 numbers from [1,2,3,4]
    job_assign_machine[i] = np.random.choice(range(1, machine_num + 1), np.random.randint(1, 5), replace=False)
    job_assign_machine[i].sort()
    print(job_assign_machine[i])

# 遍历job_assign_machine
print('equal constraint')
for i in range(job_num):
    cmd = 'lbp += ('
    for j in range(len(job_assign_machine[i])):
        cmd += f'x{i + 1}{job_assign_machine[i][j]}'
        # 如果j不是最后一个元素，就加上+
        if j != len(job_assign_machine[i]) - 1:
            cmd += ' + '
    cmd += f' == {weight[i]})'
    exec(cmd)
    print(cmd)

print('not equal constraint')
for i in range(machine_num):
    cmd = 'lbp += ('
    for j in range(len(job_assign_machine)):
        if i + 1 in job_assign_machine[j]:
            cmd += f'x{j + 1}{i + 1} + '
    if cmd == 'lbp += (':
        continue
    cmd = cmd[:-3]
    cmd += f' <= L)'
    exec(cmd)
    print(cmd)

lbp.solve()
print(lbp.name)  # 输出求解状态
print("Status:", pulp.LpStatus[lbp.status])  # 输出求解状态
for v in lbp.variables():
    print(v.name, "=", int(v.varValue))  # 输出每个变量的最优值
print("F1(x)=", pulp.value(lbp.objective))  # 输出最优解的目标函数值

# 使用nx画6个job的依赖图
G = nx.DiGraph()
for i in range(job_num):
    for j in range(len(job_assign_machine[i])):
        if j == 0:
            G.add_edge(0, i + 1)
        else:
            G.add_edge(job_assign_machine[i][j - 1], i + 1)
nx.draw(G, with_labels=True)
plt.show()
