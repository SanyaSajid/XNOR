import numpy as np
# Sigmoid activation function
def sigmoid(z):
 return 1 / (1 + np.exp(-z))
# Function to initialize parameters (weights and biases)
def initialize_parameters(nx, n_h, n_y):
 W1 = np.random.randn(n_h, nx) * 0.01 # Weights for the hidden layer
 b1 = np.zeros((n_h, 1)) # Biases for the hidden layer
 W2 = np.random.randn(n_y, n_h) * 0.01 # Weights for the output layer
 b2 = np.zeros((n_y, 1)) # Biases for the output layer
 parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
 return parameters
# Forward propagation function
def forward_propagation(X, parameters):
 W1 = parameters["W1"]
 b1 = parameters["b1"]
 W2 = parameters["W2"]
 b2 = parameters["b2"]
 Z1 = np.dot(W1, X) + b1 # Linear activation for hidden layer
 A1 = sigmoid(Z1) # Activation for hidden layer
 Z2 = np.dot(W2, A1) + b2 # Linear activation for output layer
 A2 = sigmoid(Z2) # Activation for output layer (final prediction)
 cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
 return A2, cache
# Compute cost (cross-entropy loss)
def compute_cost(A2, Y, m):
 logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
 cost = -np.sum(logprobs) / m
 return cost
# Backward propagation function to compute gradients
def backward_propagation(parameters, cache, X, Y):
 m = X.shape[1]
 W2 = parameters["W2"]
 A1 = cache["A1"]
 A2 = cache["A2"]
 dZ2 = A2 - Y
 dW2 = np.dot(dZ2, A1.T) / m
 db2 = np.sum(dZ2, axis=1, keepdims=True) / m
 dA1 = np.dot(W2.T, dZ2)
 dZ1 = dA1 * A1 * (1 - A1)
 dW1 = np.dot(dZ1, X.T) / m
 db1 = np.sum(dZ1, axis=1, keepdims=True) / m
 gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
 return gradients
# Update parameters using gradient descent
def update_parameters(parameters, gradients, learning_rate):
 W1 = parameters["W1"] - learning_rate * gradients["dW1"]
 b1 = parameters["b1"] - learning_rate * gradients["db1"]
 W2 = parameters["W2"] - learning_rate * gradients["dW2"]
 b2 = parameters["b2"] - learning_rate * gradients["db2"]
 parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
 return parameters
# Neural network model to train the XNOR function
def nn_model(X, Y, n_h, n_y, num_of_iters, learning_rate):
 nx = X.shape[0] # Number of input features
 m = X.shape[1] # Number of examples
 parameters = initialize_parameters(nx, n_h, n_y) # Initialize parameters
 for i in range(num_of_iters):
 # Forward propagation
    A2, cache = forward_propagation(X, parameters)
 # Compute cost
    cost = compute_cost(A2, Y, m)
 # Backward propagation
    gradients = backward_propagation(parameters, cache, X, Y)
 # Update parameters
    parameters = update_parameters(parameters, gradients, learning_rate)
    if i % 100 == 0:
      print(f"Iteration {i}, Cost: {cost:.4f}")
    return parameters
# Prediction function based on forward propagation
def predict(X, parameters):
 A2, _ = forward_propagation(X, parameters)
 predictions = (A2 > 0.5) * 1.0 # Thresholding predictions
 return predictions
# Input data for XNOR gate (4 training examples)
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # Inputs
Y = np.array([[1, 0, 0, 1]]) # XNOR outputs
# Set hyperparameters
n_h = 2 # Number of hidden neurons
n_y = 1 # Number of output neurons
num_of_iters = 1000
learning_rate = 0.3
# Train the neural network
trained_parameters = nn_model(X, Y, n_h, n_y, num_of_iters, learning_rate)
# Test input to calculate XNOR of its elements
X_test = np.array([[1], [1]]) # Example: XNOR of (1,1) should be 1
y_predict = predict(X_test, trained_parameters)
# Output the prediction
print(f"Neural Network prediction for XNOR example ({X_test[0][0]}, {X_test[1][0]}) is
{int(y_predict[0][0])}")
