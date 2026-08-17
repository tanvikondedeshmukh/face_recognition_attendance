# Smart Face Recognition Attendance System

An automated attendance management system using OpenCV and deep-learning-based facial recognition. It includes both a **Tkinter Graphical User Interface (GUI)** for real-time user enrollment/login and a **CLI mode** for continuous webcam monitoring.

---

## Features

- **Tkinter GUI Desktop Application**:
  - Live webcam feed integration.
  - User registration module (captures user photos directly via webcam).
  - One-click facial login and automated logging.
- **Continuous CLI Mode**: Real-time video processing with dynamic visual bounding boxes.
- **Attendance Logging**: Automatically creates and updates `Attendance.csv` with unique daily entries (Name, Date, Time).

---

## Directory Structure

```text
face-recognition-attendance/
│
├── known_faces/          # Store training images for CLI system (main.py)
│   └── .gitkeep
├── db/                   # Store registered user images from GUI app
│   └── .gitkeep
├── .gitignore            # Git ignore file
├── Attendance.csv        # Logged attendance records (generated at runtime)
├── gui_app.py            # Main Graphical Interface App (Tkinter)
├── main.py               # Standalone / CLI Real-Time Attendance Tracker
├── README.md             # Project documentation for GitHub
├── requirements.txt      # Python dependencies
└── util.py               # GUI helpers and face recognition utility functions
```

---

## Installation & Setup

### 1. Prerequisites
Ensure you have C++ compilation tools installed (required by `dlib`):
- **Windows**: Install Visual Studio with C++ Desktop Development tools.
- **macOS / Linux**: Install `cmake`:
  ```bash
  brew install cmake
  ```

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-attendance.git
cd face-recognition-attendance

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## How to Run

### GUI Interface (Recommended)
```bash
python gui_app.py
```
1. Click **Register New User**, enter the name, and confirm to store the user in `./db`.
2. Click **Login** to perform facial recognition and log attendance to `Attendance.csv`.

### Continuous Webcam CLI Tracker
```bash
python main.py
```
- Place reference photos in the `known_faces/` folder named as `Person_Name.jpg`.
- Press `q` to terminate the stream.
