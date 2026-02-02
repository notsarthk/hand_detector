import cv2
import numpy as np
import time
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── Modern MediaPipe Setup ───
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

MODEL_PATH = "hand_landmarker.task"  # Must be in same folder!

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55
    )
    landmarker = HandLandmarker.create_from_options(options)
except Exception as e:
    print("HandLandmarker failed to load:")
    print("→ Ensure 'hand_landmarker.task' is downloaded and in this folder.")
    print("→ Link: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    print(f"Error: {e}")
    exit()

# ─── Face Cascade ───
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── Hand connections for drawing ───
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),     # thumb
    (0,5), (5,6), (6,7), (7,8),     # index
    (5,9), (9,13), (13,17),         # palm cross
    (9,10),(10,11),(11,12),         # middle
    (13,14),(14,15),(15,16),        # ring
    (17,18),(18,19),(19,20),        # pinky
    (0,17)                          # palm base
]

# ─── Helper: Euclidean distance between two landmarks ───
def dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# ─── Improved & more accurate gesture detection ───
def detect_gesture(landmarks, handedness):
    if not landmarks or len(landmarks) != 21:
        return "No hand", 0.0

    wrist = landmarks[0]
    thumb_mcp = landmarks[1]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    index_tip = landmarks[8]

    middle_mcp = landmarks[9]
    middle_pip = landmarks[10]
    middle_tip = landmarks[12]

    ring_mcp = landmarks[13]
    ring_pip = landmarks[14]
    ring_tip = landmarks[16]

    pinky_mcp = landmarks[17]
    pinky_pip = landmarks[18]
    pinky_tip = landmarks[20]

    # ─── Finger extended checks (tip farther from MCP than PIP is curled) ───
    # For thumb: compare horizontal/vertical offset + distance
    thumb_extended = dist(thumb_tip, wrist) > dist(thumb_ip, wrist) * 1.3 or thumb_tip.x > thumb_ip.x + 0.05

    index_extended = dist(index_tip, index_mcp) > dist(index_pip, index_mcp) * 1.4
    middle_extended = dist(middle_tip, middle_mcp) > dist(middle_pip, middle_mcp) * 1.4
    ring_extended = dist(ring_tip, ring_mcp) > dist(ring_pip, ring_mcp) * 1.4
    pinky_extended = dist(pinky_tip, pinky_mcp) > dist(pinky_pip, pinky_mcp) * 1.4

    extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])

    confidence = 0.0
    gesture = "Unknown 🤔"

    # ─── More robust rules ───
    if thumb_extended and extended_count == 0:
        gesture = "Thumbs Up 👍"
        confidence = 0.95
    elif extended_count == 2 and index_extended and middle_extended and not ring_extended and not pinky_extended:
        # Extra check: tips roughly same height
        if abs(index_tip.y - middle_tip.y) < 0.05:
            gesture = "Peace / Victory ✌️"
            confidence = 0.92
    elif extended_count == 0 and not thumb_extended:
        gesture = "Fist ✊"
        confidence = 0.90
    elif thumb_extended and index_extended and extended_count == 1:
        # OK: thumb and index close together
        if dist(thumb_tip, index_tip) < 0.08:
            gesture = "OK 👌"
            confidence = 0.88
    elif extended_count == 4 and thumb_extended:
        gesture = "High Five / Stop ✋"
        confidence = 0.90
    elif extended_count == 4 and not thumb_extended:
        gesture = "Open Palm ✋"
        confidence = 0.85
    elif extended_count == 1 and index_extended and not thumb_extended:
        gesture = "Pointing Up 👆"
        confidence = 0.87
    elif extended_count == 3 and index_extended and middle_extended and ring_extended and not pinky_extended:
        gesture = "Three 3️⃣"
        confidence = 0.82

    # Hand side
    side = handedness[0].category_name if handedness and len(handedness) > 0 else "?"

    return f"{side} hand: {gesture} ({confidence:.0%})", confidence

# ─── Main Loop ───
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not opening!")
    exit()

print("\n=== More Accurate Gesture & Face Detector ===")
print("Improved rules: better rotation handling, confidence score, connections drawn")
print("Try: Thumbs Up, Peace, Fist, OK, Palm Open, Pointing, Stop...")
print("Press 'q' to quit\n")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Hands
    results = landmarker.detect(mp_image)

    gestures = []

    if results.hand_landmarks:
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            # Draw dots
            for lm in hand_landmarks:
                px = int(lm.x * frame.shape[1])
                py = int(lm.y * frame.shape[0])
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)

            # Draw connections (blue lines)
            for conn in HAND_CONNECTIONS:
                p1 = hand_landmarks[conn[0]]
                p2 = hand_landmarks[conn[1]]
                x1 = int(p1.x * frame.shape[1])
                y1 = int(p1.y * frame.shape[0])
                x2 = int(p2.x * frame.shape[1])
                y2 = int(p2.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            handedness = results.handedness[i] if i < len(results.handedness) else None
            gesture_text, conf = detect_gesture(hand_landmarks, handedness)
            gestures.append(gesture_text)

    # Display
    y_pos = 60
    for txt in gestures or ["No hand detected"]:
        color = (0, 255, 255) if "Unknown" not in txt else (100, 100, 255)
        cv2.putText(frame, txt, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)
        y_pos += 50

    # FPS
    fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Improved Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()