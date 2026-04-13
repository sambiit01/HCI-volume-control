import cv2
import time
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
import os

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class NumpyMLP:
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.w1, self.b1 = data['w1'], data['b1']
        self.w2, self.b2 = data['w2'], data['b2']
        self.w3, self.b3 = data['w3'], data['b3']
        
    def predict(self, x):
        x = np.array(x)
        h1 = relu(np.dot(x, self.w1) + self.b1)
        h2 = relu(np.dot(h1, self.w2) + self.b2)
        out = softmax(np.dot(h2, self.w3) + self.b3)
        return out

try:
    model = NumpyMLP("gesture_weights.npz")
    print("Loaded Numpy lightweight ML model successfully!")
except Exception as e:
    print(f"Could not load ML model: {e}")
    model = None
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    # Camera setup
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(3, wCam)
    cap.set(4, hCam)

    # MediaPipe Hand track setup
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.7, max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    # Pycaw audio control setup
    try:
        devices = AudioUtilities.GetSpeakers()
        volume = devices.EndpointVolume
        volRange = volume.GetVolumeRange()
        minVol = volRange[0]
        maxVol = volRange[1]
    except Exception as e:
        print(f"Failed to initialize pycaw: {e}")
        return
    
    vol = 0
    volBar = 400
    volPer = 0

    is_locked = False
    pinky_was_up = False
    last_toggle_time = 0

    print("Executing Hand Gesture Volume Control.")
    print("Hold up your hand and change the distance between your thumb and index finger to control the volume.")
    print("Press 'q' to quit the application.")

    while True:
        success, img = cap.read()
        if not success:
            break
            
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if len(lmList) != 0:
            # Thumb tip: id 4, Index tip: id 8
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            # Hand range roughly 50 to 150 for higher sensitivity
            vol = np.interp(length, [50, 150], [minVol, maxVol])
            volBar = np.interp(length, [50, 150], [400, 150])
            volPer = np.interp(length, [50, 150], [0, 100])
            
            # Lock toggle feature: dynamically choose between ML Classifier or math heuristic
            if model is not None:
                wrist_x = results.multi_hand_landmarks[0].landmark[0].x
                wrist_y = results.multi_hand_landmarks[0].landmark[0].y
                features = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    features.extend([lm.x - wrist_x, lm.y - wrist_y])
                prediction = model.predict([features])
                class_id = np.argmax(prediction[0])
                
                pinky_is_up = (class_id == 1)
                status_text = ["Control", "Lock", "Idle"][class_id]
                prob = prediction[0][class_id] * 100
                cv2.putText(img, f"AI: {status_text} ({prob:.1f}%)", (380, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show all probabilities for debugging
                cv2.putText(img, f"C: {prediction[0][0]:.2f} L: {prediction[0][1]:.2f} I: {prediction[0][2]:.2f}", 
                            (380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                pinky_tip_y = lmList[20][2]
                pinky_pip_y = lmList[18][2]
                pinky_is_up = pinky_tip_y < (pinky_pip_y - 20)
            
            if pinky_is_up and not pinky_was_up:
                current_time = time.time()
                if current_time - last_toggle_time > 1.0:
                    is_locked = not is_locked
                    last_toggle_time = current_time
            pinky_was_up = pinky_is_up
            
            try:
                if not is_locked:
                    volume.SetMasterVolumeLevel(vol, None)
                    if length < 50:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                else:
                    # Red circle indicates locked state
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            except Exception as e:
                pass

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 3)

        cv2.imshow("Hand Gesture Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
