## DEEPTRACE IS UNDER REVAMP.

# DeepTrace

A deepfake video detection system that analyses video content frame-by-frame, flags manipulated segments, and explains *why* each section was detected as suspicious — making the detection interpretable rather than a black-box output.

---

## How it works

1. User uploads a video through the web interface
2. MTCNN extracts faces from each frame
3. CV2 processes frames and feeds them through the detection model
4. The model (trained on FaceForensics++) scores each frame for manipulation
5. Suspicious segments are flagged with timestamp markers
6. Results are displayed with heatmap visualisations and per-frame explanations

---

## Features

- **Frame-level detection** — analyses video content frame by frame, not just a single whole-video prediction
- **Heatmap visualisation** — highlights the regions within a frame that triggered the detection
- **Per-frame explanations** — describes why each flagged segment was marked as manipulated
- **Timestamp-based flagging** — pinpoints exact segments in the video rather than a binary yes/no result
- **Web interface** — upload a video and view results directly in the browser

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Detection | Self-trained model, MTCNN, OpenCV (CV2) |
| Training data | FaceForensics++ (via Kaggle) |
| Frontend | Vanilla JS (React migration in progress) |

---

## Installation

Will be added soon!

---

## Usage

1. Open the web interface
2. Upload a video file (MP4 recommended)
3. Wait for the analysis to complete
4. View the results — flagged timestamps, heatmaps, and per-frame reasoning

---

## Model

The detection model was trained from scratch on the **FaceForensics++** dataset, which contains both pristine and manipulated videos across multiple deepfake generation methods. MTCNN is used for face detection and extraction prior to classification.

---

## Project status

Core detection pipeline is functional. Frontend UI tweaks and user authentication are currently in progress.

---

## Author

**Aman Goel**  
[LinkedIn](https://www.linkedin.com/in/aman-goel2203/) · [GitHub](https://github.com/Amangoel22)
