import time
import threading
import collections
import platform
import os
import math
import tkinter as tk
from tkinter import messagebox

import cv2
import mediapipe as mp
import winsound
import numpy as np
from flask import Flask, jsonify

# --------- CONFIG ----------
EAR_THRESHOLD = 0.25
BLINK_MAX_CLOSED = 0.8
BLINK_MIN_CLOSED = 0.03
CONSECUTIVE_WINDOW = 2.0
REQUIRED_BLINKS = 5
ONE_SEC_ALERT = 1.0
THREE_SEC_ALARM = 3.0
CAMERA_INDEX = 0

# distance threshold
DISTANCE_PIXEL_THRESHOLD = 90   # approx 80–100 cm
# ----------------------------

state = {
    "blink_count_total": 0,
    "recent_blinks": collections.deque(),
    "sleep_mode": False,
    "last_alert": "",
    "eye_closed": False,
    "closure_start": None,
    "one_sec_printed": False,
    "three_sec_triggered": False,
    "alarm_playing": False,
}

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_face_detector = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detector = mp_face_detector.FaceDetection(model_selection=0, min_detection_confidence=0.5)

RIGHT = [33, 160, 158, 133, 153, 144]
LEFT  = [362, 385, 387, 263, 373, 380]


# ---------- UTILS ----------
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts

    vert1 = euclid(p2, p6)
    vert2 = euclid(p3, p5)
    hor = euclid(p1, p4)

    if hor == 0:
        return 0.0

    return (vert1 + vert2) / (2.0 * hor)


# ----- ALARM SOUND -----
def play_alarm_once():
    if state["alarm_playing"]:
        return
    state["alarm_playing"] = True

    try:
        for _ in range(3):
            winsound.Beep(1000, 600)
    except Exception as e:
        print("Alarm error:", e)

    state["alarm_playing"] = False


# ----- SHUTDOWN CONFIRMATION -----
def confirm_shutdown():
    root = tk.Tk()
    root.withdraw()

    answer = messagebox.askyesno("Confirm Shutdown", "Do you really want to shut down the puck?")

    root.destroy()

    if answer:
        print("Shutting down system...")
        os.system("shutdown /s /t 1")
    else:
        print("Shutdown cancelled.")


# ---------- MAIN DETECTOR LOOP ----------
def detector_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    prev_eye_closed = False
    closed_since = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---------- FACE DISTANCE DETECTION ----------
        face_results = face_detector.process(rgb)

        if face_results.detections:
            detection = face_results.detections[0]
            box = detection.location_data.relative_bounding_box

            face_width_pixels = int(box.width * w)

            # Draw box
            x, y = int(box.xmin * w), int(box.ymin * h)
            fw = int(box.width * w)
            fh = int(box.height * h)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 255), 2)

            ### DISTANCE CODE ADDED ###
            if face_width_pixels < DISTANCE_PIXEL_THRESHOLD:
                print("please come a little closer")

        # ---------- EYE BLINK DETECTION ----------
        results = face_mesh.process(rgb)
        ear = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear_left = eye_aspect_ratio(lm, LEFT, w, h)
            ear_right = eye_aspect_ratio(lm, RIGHT, w, h)
            ear = (ear_left + ear_right) / 2.0

            cv2.putText(frame, f"EAR:{ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        is_closed = ear is not None and ear < EAR_THRESHOLD
        now = time.time()

        # OPEN → CLOSE
        if is_closed and not prev_eye_closed:
            closed_since = now
            state["closure_start"] = closed_since
            state["one_sec_printed"] = False
            state["three_sec_triggered"] = False

        # CLOSE → OPEN
        if (not is_closed) and prev_eye_closed:
            dur = now - closed_since

            if BLINK_MIN_CLOSED < dur < BLINK_MAX_CLOSED:
                state["blink_count_total"] += 1
                state["recent_blinks"].append(now)

                while state["recent_blinks"] and now - state["recent_blinks"][0] > CONSECUTIVE_WINDOW:
                    state["recent_blinks"].popleft()

                print(f"Your eye blink: {state['blink_count_total']}")

                if len(state["recent_blinks"]) >= REQUIRED_BLINKS and not state["sleep_mode"]:
                    print("SHUTDOWN TRIGGERED (5 blinks in 2 sec)")
                    state["sleep_mode"] = True
                    threading.Thread(target=confirm_shutdown, daemon=True).start()

            state["one_sec_printed"] = False
            state["three_sec_triggered"] = False

        # STILL CLOSED
        if is_closed:
            dur = now - closed_since

            if dur >= ONE_SEC_ALERT and not state["one_sec_printed"]:
                print("open your eyes")
                state["one_sec_printed"] = True

            if dur >= THREE_SEC_ALARM and not state["three_sec_triggered"]:
                print("ALARM TRIGGERED (eyes closed 3 sec)")
                state["three_sec_triggered"] = True
                threading.Thread(target=play_alarm_once, daemon=True).start()

        prev_eye_closed = is_closed
        state["eye_closed"] = is_closed

        cv2.imshow("Eye monitor (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- FLASK API ----------
app = Flask(__name__)

@app.route("/status")
def status():
    return jsonify({
        "blink_count_total": state["blink_count_total"],
        "recent_blinks": list(state["recent_blinks"]),
        "sleep_mode": state["sleep_mode"],
        "eye_closed": state["eye_closed"],
        "closure_start": state["closure_start"],
    })


# ---------- MAIN ----------
if __name__ == "__main__":
    threading.Thread(target=detector_loop, daemon=True).start()
    print("Flask running: http://127.0.0.1:5000/status")
    app.run(debug=False, use_reloader=False)
