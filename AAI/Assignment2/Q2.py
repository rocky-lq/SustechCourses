import numpy as np
from sklearn.metrics import mean_squared_error


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


input = np.array([0.10, 0.15])

hidden_layer_weights = np.array([[0.20, 0.30], [0.15, 0.25]])
hidden_layer_bias = 0.20
output_layer_bias = 0.40

b = np.array([hidden_layer_bias, output_layer_bias])

output_layer_weights = np.array([[0.40, 0.60], [0.35, 0.10]])

ground_truth = np.array([0.99, 0.01])

hidden_layer_output = sigmoid(np.dot(hidden_layer_weights, input) + hidden_layer_bias)
print('h:', hidden_layer_output)

output = sigmoid(np.dot(output_layer_weights, hidden_layer_output) + output_layer_bias)
print('out:', output)

error = mean_squared_error(ground_truth, output)
print('error:', error)

out1 = output[0]
o1 = ground_truth[0]
h1 = hidden_layer_output[0]
w5 = 0.40
eta = 0.1

dw = -eta * (out1 - o1) * out1 * (1 - out1) * h1
print('dw:', dw)
w5_new = w5 + dw
print('w5_new:', w5_new)
