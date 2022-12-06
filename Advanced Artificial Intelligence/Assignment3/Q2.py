import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the inputs and expected outputs
X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

# Initialize the weights randomly
w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

# Set the learning rate
learning_rate = 0.1

# Train the network for 1000 epochs
for epoch in range(1000):
    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    # Backpropagation
    layer2_error = y - layer2
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update the weights
    w1 += learning_rate * layer1.T.dot(layer2_delta)
    w0 += learning_rate * layer0.T.dot(layer1_delta)

# Calculate the accuracy
accuracy = np.mean(np.round(layer2) == y)

# Print the accuracy
print("Accuracy:", accuracy)
