import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pandas as pd

# Configuration
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "Attendance.csv"
TOLERANCE = 0.5  # Lower number = stricter matching
MODEL = "hog"    # Use 'hog' for CPU, 'cnn' for GPU/CUDA acceleration

def initialize_attendance_file():
    """Creates the Attendance.csv file with headers if it does not exist."""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

def load_known_faces():
    """Loads images from the known_faces folder and encodes them."""
    known_encodings = []
    known_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"[INFO] Created directory '{KNOWN_FACES_DIR}'. Add face images there.")
        return known_encodings, known_names

    print("[INFO] Encoding known faces...")
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(KNOWN_FACES_DIR, file)
            name = os.path.splitext(file)[0].replace("_", " ")

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"[SUCCESS] Encoded: {name}")
            else:
                print(f"[WARNING] No face found in image: {file}")

    return known_encodings, known_names

def mark_attendance(name):
    """Logs the attendance of a recognized person into Attendance.csv if not already logged today."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)

    already_marked = ((df['Name'] == name) & (df['Date'] == current_date)).any()

    if not already_marked:
        new_entry = pd.DataFrame([[name, current_date, current_time]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[ATTENDANCE MARKED] {name} at {current_time}")

def run_attendance_system():
    """Captures webcam stream, recognizes faces, and logs attendance."""
    initialize_attendance_file()
    known_encodings, known_names = load_known_faces()

    if not known_encodings:
        print("[ERROR] No known faces found. Please add images to 'known_faces/' folder.")
        return

    video_capture = cv2.VideoCapture(0)
    print("[INFO] Starting Webcam. Press 'q' to exit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to capture frame from webcam.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    mark_attendance(name)

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow("Attendance Management System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance_system()
