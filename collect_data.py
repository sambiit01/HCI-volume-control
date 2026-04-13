import cv2
import mediapipe as mp
import csv
import os
import time

def collect_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    num_samples = 300
    filename = "gestures.csv"
    
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for i in range(21):
                header.extend([f'x_{i}', f'y_{i}'])
            writer.writerow(header)
            
    gestures = [
        ("0", "Control (Thumb & Index)"),
        ("1", "Lock (Pinky Up)"),
        ("2", "Idle (Open Hand or Fist)")
    ]
    
    for label, desc in gestures:
        samples_collected = 0
        for i in range(5, 0, -1):
            success, img = cap.read()
            if success:
                cv2.putText(img, f"Get ready for: {desc}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(img, f"Starting in {i}...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Data Collection", img)
                cv2.waitKey(1000)
                
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            while samples_collected < num_samples:
                success, img = cap.read()
                if not success: break
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    row = [label]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x - wrist_x, lm.y - wrist_y])
                    writer.writerow(row)
                    samples_collected += 1
                    cv2.putText(img, f"[{desc}] Collecting... {samples_collected}/{num_samples}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection", img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
