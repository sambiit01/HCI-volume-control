# Real-Time Hand Gesture Volume Control

A Python-based system to control Windows system volume using hand gestures captured via a webcam. This project uses MediaPipe for hand tracking, OpenCV for visualization, and Pycaw for system audio control.

## Features
- **Real-time Hand Tracking**: Uses MediaPipe to detect hand landmarks.
- **Dynamic Volume Control**: Maps the distance between your thumb and index finger to the system volume.
- **Sensitivity Tuning**: Optimized gesture range for comfortable use.
- **Lock Toggle**: A specialized locking mechanism to prevent volume fluctuation.
- **Visual Feedback**: On-screen volume bar and percentage indicator.

## Requirements
- Python 3.8+ (Tested on Python 3.12)
- Webcam
- Windows OS (Required for `pycaw`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sambiit01/HCI-volume-control.git
   cd HCI-volume-control
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script:
```bash
python main.py
```

### Controls
- **Adjust Volume**: Pinch or spread your **thumb** and **index finger** tips.
- **Lock/Unlock Volume**: Flick your **pinky finger** UP to toggle the lock. 
  - **Red Circle**: Locked (Volume won't change).
  - **Green/Pink Circle**: Active volume control.
- **Quit**: Press `q` while the camera window is active.

## Project Structure
- `main.py`: The core implementation script.
- `requirements.txt`: Pin-pointed dependencies.
- `AGENTS.md`: Internal progress tracking log.
- `.gitignore`: Standard exclusion list for Python projects.

## Technical Details
- **MediaPipe Version**: `0.10.14` (Pinned for stability on Windows Python 3.12).
- **Audio Control**: Uses the EndpointVolume API via Pycaw.
- **Debounce Logic**: 1-second timeout on the lock toggle to prevent jitter-based switching.
