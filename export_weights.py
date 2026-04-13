import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

def export():
    if not os.path.exists("gesture_model.keras"):
        print("Error: gesture_model.keras missing! Run train_model.py first.")
        return
    model = tf.keras.models.load_model("gesture_model.keras")
    weights = model.get_weights()
    np.savez("gesture_weights.npz", 
             w1=weights[0], b1=weights[1], 
             w2=weights[2], b2=weights[3], 
             w3=weights[4], b3=weights[5])
    print("Successfully exported neural network weights to gesture_weights.npz!")

if __name__ == "__main__":
    export()
