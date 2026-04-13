import csv
import numpy as np
import random

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def load_data(filename="gestures.csv"):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if not row: continue
            X.append([float(val) for val in row[1:]])
            label = int(row[0])
            one_hot = [0.0] * 3
            one_hot[label] = 1.0
            y.append(one_hot)
    return np.array(X), np.array(y)

def train():
    X, y = load_data()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    input_size, h1_size, h2_size, output_size = 42, 128, 64, 3
    
    w1 = np.random.randn(input_size, h1_size) * np.sqrt(2.0/input_size)
    b1 = np.zeros((1, h1_size))
    w2 = np.random.randn(h1_size, h2_size) * np.sqrt(2.0/h1_size)
    b2 = np.zeros((1, h2_size))
    w3 = np.random.randn(h2_size, output_size) * np.sqrt(2.0/h2_size)
    b3 = np.zeros((1, output_size))
    
    lr, epochs = 0.005, 1000
    
    for epoch in range(epochs):
        z1 = np.dot(X, w1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = relu(z2)
        z3 = np.dot(a2, w3) + b3
        probs = softmax(z3)
        dz3 = (probs - y) / len(X)
        dw3, db3 = np.dot(a2.T, dz3), np.sum(dz3, axis=0, keepdims=True)
        dz2 = np.dot(dz3, w3.T) * relu_derivative(z2)
        dw2, db2 = np.dot(a1.T, dz2), np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, w2.T) * relu_derivative(z1)
        dw1, db1 = np.dot(X.T, dz1), np.sum(dz1, axis=0, keepdims=True)
        w3 -= lr * dw3; b3 -= lr * db3; w2 -= lr * dw2; b2 -= lr * db2; w1 -= lr * dw1; b1 -= lr * db1
        if epoch % 100 == 0:
            acc = np.mean(np.argmax(probs, 1) == np.argmax(y, 1))
            print(f"Epoch {epoch}, Accuracy: {acc*100:.2f}%")
            
    np.savez("gesture_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
    print("SUCCESS: NumPy AI model (no TensorFlow) saved to gesture_weights.npz!")

if __name__ == "__main__":
    train()
