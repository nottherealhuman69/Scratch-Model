import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = '/kaggle/input/deep-learning/Folds5x2_pp.xlsx'
data = pd.read_excel(url, sheet_name='Sheet1')

# Splitting data into input (X) and output (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize the input and output data between 0 and 1
def normalize(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

X = normalize(X)
y = normalize(y.reshape(-1, 1))

# Manual Data Splitting: 72% train, 18% validation, 10% test
def manual_split(X, y, train_ratio=0.72, val_ratio=0.18, test_ratio=0.10):
    n = X.shape[0]
    indices = np.random.permutation(n)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    X_train, y_train = X[indices[:train_end]], y[indices[:train_end]]
    X_val, y_val = X[indices[train_end:val_end]], y[indices[train_end:val_end]]
    X_test, y_test = X[indices[val_end:]], y[indices[val_end:]]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Splitting the dataset
X_train, y_train, X_val, y_val, X_test, y_test = manual_split(X, y)

# He initialization for weights
def he_init(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

# Define the ANN class
class ANN:
    def __init__(self, layers, learning_rate=0.001, l2_lambda=0.0001, momentum_beta=0.9):
        self.layers = layers
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.momentum_beta = momentum_beta
        self.weights = [he_init((layers[i], layers[i+1])) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self.tanh(net) if i < len(self.weights) - 1 else net
            activations.append(activation)
        return activations

    def backprop(self, X, y, activations):
        m = X.shape[0]
        deltas = [activations[-1] - y]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.tanh_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            l2_penalty = self.l2_lambda * self.weights[i]
            gradient_w = np.dot(activations[i].T, deltas[i]) / m + l2_penalty
            gradient_b = np.sum(deltas[i], axis=0) / m

            # Gradient clipping
            gradient_w = np.clip(gradient_w, -1, 1)
            gradient_b = np.clip(gradient_b, -1, 1)

            self.velocity_w[i] = self.momentum_beta * self.velocity_w[i] - self.learning_rate * gradient_w
            self.velocity_b[i] = self.momentum_beta * self.velocity_b[i] - self.learning_rate * gradient_b

            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                activations = self.forward(X_batch)
                self.backprop(X_batch, y_batch, activations)

            # Calculate losses after each epoch
            train_loss = np.mean(np.square(self.forward(X)[-1] - y))
            val_loss = np.mean(np.square(self.forward(X_val)[-1] - y_val))

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)[-1]

# Function to run experiments
def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, epochs, batch_size, learning_rate):
    print(f"\nTraining for {epochs} epochs, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    input_dim = X_train.shape[1]
    output_dim = 1

    ann = ANN(layers=[input_dim, 40, 40, 40, output_dim], learning_rate=learning_rate)
    train_losses, val_losses = ann.train(X_train, y_train, X_val, y_val, epochs, batch_size)

    # Validation
    y_val_pred = ann.predict(X_val)
    val_mape = np.mean(np.abs((y_val_pred - y_val) / (y_val + 1e-10))) * 100

    # Testing
    y_test_pred = ann.predict(X_test)
    test_mape = np.mean(np.abs((y_test_pred - y_test) / (y_test + 1e-10))) * 100

    print(f"Validation MAPE: {val_mape}")
    print(f"Test MAPE: {test_mape}")

    return {
        'epochs': epochs,
        'val_mape': val_mape,
        'test_mape': test_mape,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }

# Run experiment with different configurations
result = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, epochs=1000, batch_size=256, learning_rate=0.0001)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(result['train_losses'], label="Train Loss")
plt.plot(result['val_losses'], label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(result['y_test'], result['y_test_pred'], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Print summary of results
print("\nSummary of Results:")
print(f"Epochs: {result['epochs']}, Val MAPE: {result['val_mape']:.2f}, Test MAPE: {result['test_mape']:.2f}")
