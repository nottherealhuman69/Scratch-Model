import numpy as np
import matplotlib.pyplot as plt
class ANN:
    def __init__(self, layers, activation, learning_rate, momentum=0.9, l2_lambda=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_lambda = l2_lambda
        
        # Initialize weights and biases
        self.weights = [np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2.0 / self.layers[i-1]) 
                        for i in range(1, len(self.layers))]
        self.biases = [np.zeros((self.layers[i], 1)) for i in range(1, len(self.layers))]
        
        # Initialize velocity (for momentum)
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
        # Set the activation function
        if activation == 'tanh':
            self.activation = np.tanh
            self.activation_deriv = lambda z: 1 - np.tanh(z)**2
        elif activation == 'sigmoid':
            self.activation = lambda z: 1 / (1 + np.exp(-z))
            self.activation_deriv = lambda z: self.activation(z) * (1 - self.activation(z))
        elif activation == 'relu':
            self.activation = lambda z: np.maximum(0, z)
            self.activation_deriv = lambda z: (z > 0).astype(float)

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, self.a_values[-1]) + b
            a = self.activation(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return self.a_values[-1]

    def backward(self, X, Y):
        m = X.shape[1]  # Number of samples
        dz = self.a_values[-1] - Y
        
        grads_w = []
        grads_b = []
        
        for l in range(len(self.layers) - 1, 0, -1):
            dw = np.dot(dz, self.a_values[l-1].T) / m + (self.l2_lambda / m) * self.weights[l-1]
            db = np.sum(dz, axis=1, keepdims=True) / m
            
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
            if l > 1:
                dz = np.dot(self.weights[l-1].T, dz) * self.activation_deriv(self.z_values[l-2])
        
        # Update weights and biases using momentum and L2 regularization
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - (1 - self.momentum) * self.learning_rate * grads_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - (1 - self.momentum) * self.learning_rate * grads_b[i]
            
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(self, X, Y, batch_size, epochs=1000):
        m = X.shape[1]  # Number of samples
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            # Shuffle the training data
            perm = np.random.permutation(m)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]
                
                Y_pred = self.forward(X_batch)
                self.backward(X_batch, Y_batch)
            
            # Calculate loss after each epoch (including L2 regularization)
            train_loss = np.mean(np.square(self.forward(X) - Y))
            l2_regularization = 0.5 * self.l2_lambda * sum(np.sum(np.square(w)) for w in self.weights) / m
            train_loss += l2_regularization
            
            history['train_loss'].append(train_loss)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Loss: {train_loss}")
        return history
                
    def predict(self, X):
        return self.forward(X)
    
# Data generation
def generate_sin_data():
    x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_train = np.sin(x_train)
    
    x_val = np.random.uniform(-2 * np.pi, 2 * np.pi, 300)
    y_val = np.sin(x_val)
    
    return x_train.reshape(1, -1), y_train.reshape(1, -1), x_val.reshape(1, -1), y_val.reshape(1, -1)

# Normalize data between -1 and 1
def normalize_data(x, xmin, xmax):
    return 2 * (x - xmin) / (xmax - xmin) - 1

# Prepare data
x_train, y_train, x_val, y_val = generate_sin_data()
x_train_norm = normalize_data(x_train, np.min(x_train), np.max(x_train))
x_val_norm = normalize_data(x_val, np.min(x_train), np.max(x_train))

# Create ANN model
layers = [1, 25,12,9, 1]  # Input layer (1 neuron), hidden layer (10 neurons), output layer (1 neuron)
ann = ANN(layers=layers, activation='tanh', learning_rate=0.1)

# Train the model
ann.train(x_train_norm, y_train, 256,epochs=20000)

# Validation
y_pred = ann.predict(x_val_norm)

x_fine = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(1, -1)  # 1000 evenly spaced points between -2pi and 2pi
x_fine_norm = normalize_data(x_fine, np.min(x_train), np.max(x_train))  # Normalize the fine-grained x values
y_pred_fine = ann.predict(x_fine_norm)  # Get predictions f

# Sort the training data for smooth plotting
sorted_indices_train = np.argsort(x_train.flatten())
x_train_sorted = x_train.flatten()[sorted_indices_train]
y_train_sorted = y_train.flatten()[sorted_indices_train]

# Sort the predicted data for smooth plotting
sorted_indices_fine = np.argsort(x_fine.flatten())
x_fine_sorted = x_fine.flatten()[sorted_indices_fine]
y_pred_fine_sorted = y_pred_fine.flatten()[sorted_indices_fine]

# Sort validation points as well
sorted_indices_val = np.argsort(x_val.flatten())
x_val_sorted = x_val.flatten()[sorted_indices_val]
y_val_sorted = y_val.flatten()[sorted_indices_val]
y_pred_val_sorted = y_pred.flatten()[sorted_indices_val]

# Plotting the sine function approximation
plt.figure(figsize=(5, 3))

# Plot the actual smooth sine curve for the fine-grained x values
plt.plot(x_fine_sorted, np.sin(x_fine_sorted), label='Actual sin(x) (Smooth)', color='blue')

# Plot the predicted smooth curve from the ANN
plt.plot(x_fine_sorted, y_pred_fine_sorted, label='Predicted ANN (Smooth)', linestyle='dashed', color='orange')

# Adding labels and legend
plt.title("Sine Function Approximation with ANN")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()
def compute_mape(y_true, y_pred, epsilon=0.001):
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true))))*100
    return mape


# Calculate MAPE for validation data
mape_val = compute_mape(y_val, y_pred)
print(f"MAPE on Validation Data: {mape_val}%")

