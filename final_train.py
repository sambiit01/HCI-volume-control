import csv
import random
import tensorflow as tf
import numpy as np
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

def train_and_export():
    X, y = load_data()
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split = int(len(X) * 0.8)
    X_train, y_train = tf.constant(X[:split]), tf.constant(y[:split])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=0)
    
    weights = model.get_weights()
    np.savez("gesture_weights.npz", 
             w1=weights[0], b1=weights[1], 
             w2=weights[2], b2=weights[3], 
             w3=weights[4], b3=weights[5])
    print("SUCCESS: Model trained and gesture_weights.npz exported!")

if __name__ == "__main__":
    train_and_export()
