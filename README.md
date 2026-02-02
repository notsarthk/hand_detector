# Hand Gesture & Sign Language Detector

A real-time webcam-based hand gesture recognition project using **MediaPipe** (modern Tasks API) and **OpenCV**.  
Detects common hand signs like Thumbs Up, Peace/Victory, Fist, OK, Open Palm, Pointing, and more — with face detection as a bonus!

## Features
- Real-time hand landmark detection (21 points per hand)
- Recognizes 8+ common static gestures with improved accuracy
- Draws hand skeleton (dots + blue lines)
- Face detection with green bounding box
- Shows gesture name + confidence score
- FPS display
- Mirror webcam view for natural feel

## Gestures Currently Supported
- 👍 Thumbs Up
- ✌️ Peace / Victory
- ✊ Fist
- 👌 OK
- ✋ Open Palm / High Five / Stop
- 👆 Pointing Up
- 3️⃣ Three Fingers
- 🤔 Unknown (with low confidence)

## Requirements
- Python 3.8+
- Webcam

### Python Packages
```bash
pip install opencv-python mediapipe numpy


