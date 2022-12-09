import networkx as nx

# 构建有向图
G = nx.DiGraph()

# 添加源点和汇点
for i in range(1, 6):
    G.add_node('s' + str(i))
    G.add_node('t' + str(i))

# 添加边
for i in range(1, 6):
    G.add_edge('s' + str(i), 't' + str(i), capacity=2)
    G.add_edge('s' + str(i), 't' + str(i), capacity=2)

# 计算网络流
flow_value, flow_dict = nx.maximum_flow(G, 's1', 't1')

# 输出结果
print(flow_value)
print(flow_dict)
