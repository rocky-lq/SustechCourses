import numpy as np
from sklearn.metrics import mean_squared_error

input = np.array([0.10, 0.15])

hidden_layer_weights = np.array([[0.20, 0.15], [0.30, 0.25]])
hidden_layer_bias = 0.20
output_layer_bias = 0.40

b = np.array([hidden_layer_bias, output_layer_bias])

output_layer_weights = np.array([[0.40, 0.35], [0.60, 0.10]])

ground_truth = np.array([0.99, 0.01])

hidden_layer_output = np.dot(input, hidden_layer_weights) + hidden_layer_bias
print(hidden_layer_output)

output = np.dot(hidden_layer_output, output_layer_weights) + output_layer_bias
print(output)

error = mean_squared_error(ground_truth, output)
print(error)
