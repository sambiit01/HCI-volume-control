# 🖐️ AI Hand Gesture Volume Control (HCI)

A state-of-the-art Human-Computer Interaction (HCI) system that allows you to control your Windows system volume through real-time hand gestures. This project combines computer vision with custom-trained Neural Networks for a seamless, touchless experience.

---

## 🚀 Key Features
- **Precise Volume Control**: Map the distance between your thumb and index finger to the system volume in real-time.
- **AI-Powered Lock Toggle**: Use your pinky finger to "Lock" the volume, preventing accidental changes while moving your hand.
- **Dual-State Indicators**: 
  - 🟢 **Pink/Green Circle**: Active tracking mode.
  - 🔴 **Red Circle**: Locked mode (Safety ignore).
- **In-App Dashboard**: Real-time display of volume percentage, visual bar, and AI classification confidence.

---

## 🏗️ Technical Architecture & Design

### 1. Computer Vision Layer
We use **MediaPipe Hands** to extract 21 precise 3D landmarks from the webcam feed. This provides a stable skeleton regardless of lighting or background.

### 2. The "Intelligence" (Custom Neural Network)
Unlike basic mathematical scripts, this system uses a custom-trained **Multi-Layer Perceptron (MLP)**.
- **Input**: 42 normalized hand coordinates (x, y relative to the wrist).
- **Architecture**: 3-layer dense network (128 -> 64 -> 3 neurons).
- **Classes**: `Control`, `Lock (Pinky Up)`, and `Idle`.
- **Optimization**: The model was trained to **96.2% accuracy** to ensure the "Lock" gesture is never missed.

### 3. High-Performance Inference (The NumPy Bypass)
A major design challenge on Windows is that TensorFlow's C++ drivers often conflict with OpenCV/MediaPipe camera drivers. 
- **The Solution**: I developed a **Pure NumPy Inference Engine**. 
- We trained the model in TensorFlow, exported the mathematical "Weights," and wrote a lightweight neural network from scratch using just math. 
- **Result**: Zero crashes, instant startup, and 0.001s processing time.

---

## 🛠️ Toolset
- **Python 3.12**
- **OpenCV**: Video capture and UI rendering.
- **MediaPipe**: Hand landmark extraction.
- **Pycaw**: Native Windows Audio Endpoint control.
- **NumPy**: The mathematical engine for our AI brain.
- **TensorFlow**: (Used for offline training only).

---

## 📂 Project Structure
- `main.py`: The live application and NumPy inference engine.
- `gesture_weights.npz`: The pre-trained AI brain.
- `numpy_train.py`: First-principles trainer (No TF required).
- `collect_data.py`: Tool to record your own custom gesture datasets.
- `train_model.py` / `final_train.py`: TensorFlow-based training scripts.

---

## 🎮 How to Use

1. **Setup**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run**:
   ```bash
   python main.py
   ```

3. **Interact**:
   - **Volume**: Change the distance between thumb and index.
   - **Lock**: Flick your **Pinky Finger UP** once to lock. Flick it again to unlock.

---

## 📝 License
This project was developed as a Human-Computer Interaction demonstration using state-of-the-art Computer Vision and Machine Learning principles.
