import csv
import random
import tensorflow as tf
import os

# Suppress messy TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data(filename="gestures.csv"):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row: continue
            X.append([float(val) for val in row[1:]])
            y.append(int(row[0]))
    return X, y

def train():
    if not os.path.exists("gestures.csv"):
        print("Error: gestures.csv not found!")
        return
    X, y = load_data()
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    
    split = int(len(X) * 0.8)
    X_train, y_train = tf.constant(X[:split]), tf.constant(y[:split])
    X_val, y_val = tf.constant(X[split:]), tf.constant(y[split:])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))
    model.save("gesture_model.keras")
    print("SUCCESS: Model saved to gesture_model.keras")

if __name__ == "__main__":
    train()
