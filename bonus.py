import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url = '/content/Folds5x2_pp.xlsx'
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

# Define the ANN class with both SGD with momentum and Adam options
class ANN:
    def _init_(self, layers, learning_rate=0.001, l2_lambda=0.0001, momentum_beta=0.9, optimizer='sgd', adam_beta1=0.9, adam_beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.momentum_beta = momentum_beta
        self.optimizer = optimizer  # 'sgd' or 'adam'
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.epsilon = epsilon
        
        # Initialize weights, biases, and velocities for momentum or Adam
        self.weights = [he_init((layers[i], layers[i+1])) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        if optimizer == 'adam':
            self.m_t_w = [np.zeros_like(w) for w in self.weights]  # First moment estimate for weights
            self.m_t_b = [np.zeros_like(b) for b in self.biases]  # First moment estimate for biases
            self.v_t_w = [np.zeros_like(w) for w in self.weights]  # Second moment estimate for weights
            self.v_t_b = [np.zeros_like(b) for b in self.biases]  # Second moment estimate for biases
            self.t = 0  # Time step

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self.leaky_relu(net) if i < len(self.weights) - 1 else net
            activations.append(activation)
        return activations

    def backprop(self, X, y, activations):
        m = X.shape[0]
        deltas = [activations[-1] - y]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.leaky_relu_derivative(activations[i])
            deltas.append(delta)
        deltas.reverse()
        
        for i in range(len(self.weights)):
            l2_penalty = self.l2_lambda * self.weights[i]
            gradient_w = np.dot(activations[i].T, deltas[i]) / m + l2_penalty
            gradient_b = np.sum(deltas[i], axis=0) / m
            
            if self.optimizer == 'sgd':
                # SGD with Momentum
                self.velocity_w[i] = self.momentum_beta * self.velocity_w[i] - self.learning_rate * gradient_w
                self.velocity_b[i] = self.momentum_beta * self.velocity_b[i] - self.learning_rate * gradient_b
                self.weights[i] += self.velocity_w[i]
                self.biases[i] += self.velocity_b[i]
            
            elif self.optimizer == 'adam':
                # Adam Optimization
                self.t += 1
                self.m_t_w[i] = self.adam_beta1 * self.m_t_w[i] + (1 - self.adam_beta1) * gradient_w
                self.m_t_b[i] = self.adam_beta1 * self.m_t_b[i] + (1 - self.adam_beta1) * gradient_b
                self.v_t_w[i] = self.adam_beta2 * self.v_t_w[i] + (1 - self.adam_beta2) * (gradient_w ** 2)
                self.v_t_b[i] = self.adam_beta2 * self.v_t_b[i] + (1 - self.adam_beta2) * (gradient_b ** 2)
                
                # Bias correction
                m_hat_w = self.m_t_w[i] / (1 - self.adam_beta1 ** self.t)
                m_hat_b = self.m_t_b[i] / (1 - self.adam_beta1 ** self.t)
                v_hat_w = self.v_t_w[i] / (1 - self.adam_beta2 ** self.t)
                v_hat_b = self.v_t_b[i] / (1 - self.adam_beta2 ** self.t)
                
                # Update weights and biases
                self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
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
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")
        
        return train_losses, val_losses

    def predict(self, X):
        return self.forward(X)[-1]

# Function to run experiments
def run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, optimizer='sgd', epochs=100, batch_size=32, learning_rate=0.001):
    print(f"\nTraining with {optimizer.upper()} for {epochs} epochs, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    input_dim = X_train.shape[1]
    output_dim = 1
    
    ann = ANN(layers=[input_dim, 64, 32, 16, output_dim], learning_rate=learning_rate, optimizer=optimizer)
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

# Run experiments with SGD and Adam optimizers and compare
result_sgd = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, optimizer='sgd', epochs=200, batch_size=64, learning_rate=0.01)
result_adam = run_experiment(X_train, y_train, X_val, y_val, X_test, y_test, optimizer='adam', epochs=200, batch_size=64, learning_rate=0.01)

# Plot convergence history for SGD vs Adam
plt.figure(figsize=(10, 6))
plt.plot(result_sgd['train_losses'], label="SGD Train Loss")
plt.plot(result_sgd['val_losses'], label="SGD Validation Loss")
plt.plot(result_adam['train_losses'], label="Adam Train Loss", linestyle='dashed')
plt.plot(result_adam['val_losses'], label="Adam Validation Loss", linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence Comparison: SGD vs Adam')
plt.legend()
plt.show()

# Print summary of results for both optimizers
print("\nSummary of Results with SGD:")
print(f"Epochs: {result_sgd['epochs']}, Val MAPE: {result_sgd['val_mape']:.2f}, Test MAPE: {result_sgd['test_mape']:.2f}")

print("\nSummary of Results with Adam:")
print(f"Epochs: {result_adam['epochs']}, Val MAPE: {result_adam['val_mape']:.2f}, Test MAPE: {result_adam['test_mape']:.2f}")