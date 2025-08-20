import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Streamlit UI setup
st.set_page_config(page_title="Gesture Painter", layout="wide")
st.title("🎨 Gesture-based Virtual Painting Board")

# Sidebar tools
st.sidebar.header("🛠️ Tools")
brush_color = st.sidebar.color_picker("🎨 Pick Brush Color", "#FF0000")
brush_size = st.sidebar.slider("✏️ Brush Size", 5, 50, 15)
eraser_on = st.sidebar.checkbox("🧽 Use Eraser", False)

# Hand tracking setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Create canvas (whiteboard)
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
prev_x, prev_y = None, None

# Webcam capture
cap = cv2.VideoCapture(0)

stframe = st.empty()  # Streamlit live frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("❌ Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get fingertip coordinates (Index finger = landmark 8)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y

            # Eraser or brush
            if eraser_on:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 255), brush_size * 2)
            else:
                bgr = tuple(int(brush_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # HEX → BGR
                cv2.line(canvas, (prev_x, prev_y), (x, y), bgr, brush_size)

            prev_x, prev_y = x, y

            # Draw hand skeleton (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = None, None

    # Combine live video + drawing canvas
    output = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)

    # Show inside Streamlit
    stframe.image(output, channels="BGR")

cap.release()
