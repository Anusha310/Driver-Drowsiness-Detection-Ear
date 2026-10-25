# 🚗 Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System built using Python, OpenCV, and MediaPipe. The application monitors facial landmarks to estimate eye closure, detect yawning, and identify signs of driver fatigue. It provides visual feedback and audio alerts to help warn drivers when drowsiness is detected.

## 📌 Project Overview

Driver drowsiness is a significant road safety concern. This project aims to detect potential signs of driver fatigue through real-time facial landmark analysis.

The application processes webcam or video input, calculates Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR), monitors head pose, and maintains a drowsiness score. Based on the detection state, it displays a visual dashboard and manages alerts.

## ✨ Features

- **Real-Time Face Monitoring:** Processes webcam input using OpenCV and MediaPipe Face Landmarker.
- **Eye Aspect Ratio (EAR):** Estimates eye openness to help identify prolonged eye closure.
- **Yawning Detection:** Uses Mouth Aspect Ratio (MAR) to identify potential yawning.
- **Head Pose Monitoring:** Estimates head orientation to account for possible signs of fatigue.
- **Personalized EAR Calibration:** Establishes an individual eye-openness baseline during initial processing.
- **Drowsiness Scoring:** Tracks evidence of potential drowsiness over time using a state-based scoring system.
- **Multi-Level Alerts:** Supports visual warnings and audio alerts based on the detection state.
- **Interactive Dashboard:** Displays detection status, EAR, MAR, confidence information, and session statistics.
- **Session Logging:** Records events and generates a session summary.
- **Video Input Support:** Supports webcam input and video files for testing.

## 🛠️ Technologies Used

- **Python** — Application logic and integration.
- **OpenCV** — Video capture, frame processing, and display.
- **MediaPipe** — Facial landmark detection.
- **NumPy** — Numerical calculations and array processing.
- **Pygame** — Audio alert playback.
- **TensorFlow/Keras** — Optional model-loading support in the application.

## 📂 Project Structure

```text
Driver-Drowsiness-Detection-Ear/
│
├── main.py                  # Application entry point
├── detection_engine.py      # Drowsiness scoring and state management
├── ear_calculator.py       # EAR, MAR, head pose, and calibration
├── alert_manager.py        # Visual and audio alert management
├── ui_overlay.py           # On-screen dashboard and visual indicators
├── utils.py                 # Session logging and summary generation
├── requirements.txt        # Python dependencies
├── face_landmarker.task    # MediaPipe face landmark model
├── alert.mp3                # Audio alert file
└── .gitignore               # Files excluded from Git
```

## ⚙️ Installation

### Prerequisites

- Python installed on your computer.
- Git installed if you want to clone the repository.
- A webcam for real-time detection.
- A Windows, Linux, or macOS environment with the required dependencies supported.

### 1. Clone the repository

```bash
git clone https://github.com/Anusha310/Driver-Drowsiness-Detection-Ear.git
```

### 2. Navigate to the project directory

```bash
cd Driver-Drowsiness-Detection-Ear
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** TensorFlow is included in the requirements file. Depending on your Python version and operating system, you may need a compatible TensorFlow version.

## ▶️ How to Run

### Run with your webcam

```bash
python main.py
```

The application uses webcam index `0` by default.

### Run with a video file

```bash
python main.py --video path/to/your/video.mp4
```

Replace `path/to/your/video.mp4` with the path to a video file on your computer.

### Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `S` | Stop the current alert and reset the detection state |

## 🧠 How It Works

1. **Video Capture:** OpenCV captures frames from a webcam or video file.
2. **Face Landmark Detection:** MediaPipe identifies facial landmarks.
3. **Eye Analysis:** The system calculates EAR to estimate whether the eyes are open or closed.
4. **Mouth Analysis:** MAR is calculated to help identify yawning.
5. **Head Pose Estimation:** Facial landmark positions are used to estimate head orientation.
6. **Calibration:** The system collects initial open-eye measurements to personalize the EAR threshold.
7. **Drowsiness Scoring:** The detection engine updates its score using eye-closure duration and other implemented signals.
8. **Alert Management:** The application changes its detection state and manages visual or audio alerts.
9. **Session Logging:** Events and session statistics are recorded for review.

## 📊 Detection States

The application maintains three primary detection states:

- **Awake:** No current high-level drowsiness alert.
- **Warning:** Signs of possible drowsiness require attention.
- **Alert:** The detection score has reached the configured alert threshold.

The system also displays relevant measurements and status information through its dashboard.

## 🔮 Future Improvements

Potential improvements include:

- Improving detection performance under different lighting conditions.
- Evaluating detection accuracy across different users and environments.
- Integrating and validating CNN predictions within the drowsiness scoring logic.
- Adding more robust head pose estimation.
- Improving configuration options for thresholds and alert sensitivity.
- Adding automated tests and performance evaluation.
- Testing the system across a wider range of cameras and hardware.

## ⚠️ Limitations and Safety Notice

This project is intended for educational and demonstration purposes. Its results may be affected by lighting, camera placement, facial characteristics, and other environmental conditions.

It is not a certified driver safety system and should not replace adequate rest, responsible driving, or established road safety practices.

## 👩‍💻 Author

**Anusha310**

GitHub: [Anusha310](https://github.com/Anusha310)

## 📄 License

No license has been specified for this repository yet. Unless a license is added, the project should not be assumed to grant permission for reuse, modification, or redistribution.
