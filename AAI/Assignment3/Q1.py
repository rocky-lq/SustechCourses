import numpy as np

# Define the input data for the XOR operation

x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

# Define the structure of the ANN
input_size = 3
hidden_size = 3
output_size = 1

# Initialize the weights of the network using random values
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(1, hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(1, output_size)


# Define the activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Define the derivative of the activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Train the network using backpropagation
for epoch in range(10000):
    # Forward propagation
    Z1 = np.dot(x, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Calculate the error
    error = y - A2

    # Backpropagation
    dZ2 = error * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(x.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update the weights
    W1 += dW1
    b1 += db1
    W2 += dW2
    b2 += db2

# Test the network on new data
x_test = x

Z1 = np.dot(x_test, W1) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2)

# 对A2四舍五入

print(np.round(A2))

# 对比A2与y的值，计算准确率
accuracy = np.mean(np.round(A2) == y)
print("Accuracy:", accuracy)

# 保留两位小数
np.set_printoptions(precision=2)
print(W1)
print(b1)
print(W2)
print(b2)

# 可视化该神经网络，将权重显示在图中
import matplotlib.pyplot as plt
import networkx as nx


# 定义一个函数，用于绘制神经网络
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)

    # for n, layer_size in enumerate([3, 3, 1]):
    #     layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
    #     for m in range(layer_size):
    #         if n == 0:
    #             ax.text(-0.1, layer_top - m * v_spacing, round(W1[0][m], 2), fontsize=14)
    #             ax.text(-0.1, layer_top - m * v_spacing - 0.05, round(W1[1][m], 2), fontsize=14)
    #             ax.text(-0.1, layer_top - m * v_spacing - 0.1, round(W1[2][m], 2), fontsize=14)
    #             ax.text(-0.1, layer_top - m * v_spacing - 0.15, round(b1[0][m], 2), fontsize=14)
    #         elif n == 1:
    #             ax.text(0.9, layer_top - m * v_spacing, round(W2[m][0], 2), fontsize=14)
    #             ax.text(0.85, layer_top - m * v_spacing, round(b2[0][0], 2), fontsize=14)


# 绘制神经网络
fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')

draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, [3, 3, 1])

# 显示权重
plt.show()

# 保存图片
fig.savefig('neural_network.png')
