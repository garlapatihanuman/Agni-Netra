# 🔥 AgniNetra — Next-Gen Tactical Fire, Live Flame & Smoke Detection

AgniNetra is a real-time computer vision command platform powered by YOLOv8 and optical flame thermal analysis. It features an interactive **Three.js 3D WebGL Holographic Hazard Core**, **continuous live webcam streaming**, **tactical HUD bounding boxes**, **Web Audio sirens**, and high-precision detection distinguishing **Live Flame**, **Fire**, and **Smoke**.

---

## 🌟 Key Features

1. **Interactive 3D WebGL Hologram Core**:
   - Over 1,800 dynamic particles simulating flame vortices, rising embers, and volumetric smoke.
   - Interactive 3D mouse rotation and orbital scanner rings.
   - Real-time hazard state synchronizing with detections (`SURVEILLANCE`, `LIVE FLAME`, `INFERNO`, `SMOKE`).

2. **High-Precision Neural Detection (Powered by `weights/best.pt`)**:
   - Exclusively powered by the trained YOLOv8 neural network (`weights/best.pt`).
   - Analyzes spatial gradients, textures, boundaries, and combustion patterns to **completely eliminate false positives on orange clothes, walls, and ambient warm lighting**.
   - Differentiates:
     - **🔥 LIVE FLAME**: Compact luminous flames (lighters, candles, matches, torches, burners).
     - **🌋 FIRE**: Large open fires, bonfires, and structural blazes.
     - **💨 SMOKE**: Smoke plumes, billowing haze, and particulate dispersion.

3. **Ultra-Responsive Live Webcam Streaming**:
   - Real-time continuous OpenCV camera stream with instant start/stop controls.
   - Real-time FPS counter, inference latency in milliseconds, and active hazard radar.
   - Browser snapshot fallback mode for single-frame inspections.

4. **Modern Cyberpunk & Glassmorphism Command Center**:
   - Deep midnight dark theme with neon hazard accents (`#ff6a00`, `#ff1744`, `#00f0ff`).
   - Frosted glass cards, pulsating threat level banners, and tactical targeting reticles.
   - Emergency Web Audio synthesizer alarm siren (with Mute/Unmute toggle).

5. **High-Precision Image Inspection & Upload**:
   - Dedicated drag-and-drop surveillance imagery upload (PNG, JPG, JPEG, WEBP).
   - Real-time side-by-side comparison between raw surveillance source and tactical AI annotations.
   - Comprehensive target classification registry with bounding coordinates and exportable results.

6. **Full Video & Multi-Source Inspection**:
   - Multi-frame video file analysis with hazard tracking progress and summary metrics.
   - Downloadable annotated inspection outputs.

---

## 📸 Project Screenshots

### Main Dashboard
![Agni-Netra Dashboard](ScreenShots/dashboard.png)

### Fire Detection
![Fire Detection](ScreenShots/fire-detection.png)

### Live Camera Detection
![Live Camera Detection](ScreenShots/live-camera-detection.png)

### Smoke Detection
![Smoke Detection](ScreenShots/smoke-detection.png)

## 🚀 Quick Start

### 1. Activate Environment & Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Model Weights
Ensure your trained YOLOv8 weights are placed at:
```text
weights/best.pt
```

### 3. Launch the Application
```bash
streamlit run frontend/app.py
```
Then navigate to `http://localhost:8501` in your web browser.
The legacy root command `streamlit run app.py` remains available as a compatibility entry point.

---

## ⚙️ Configuration (`backend/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `"weights/best.pt"` | Path to the trained YOLOv8 model weights |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence cutoff (filters noise while capturing real flames) |
| `IOU_THRESHOLD` | `0.45` | Non-Maximum Suppression (NMS) intersection threshold |
| `IMAGE_SIZE` | `640` | Inference resolution (balanced speed vs accuracy) |
| `DEVICE` | `"cpu"` | Compute accelerator (`cpu`, `mps`, or `cuda`) |
| `ENABLE_AUDIO_ALARM` | `True` | Emergency Web Audio siren upon flame/fire detection |

---

## 📁 Project Architecture

```
firedetection/
├── frontend/
│   ├── app.py          # Streamlit Command Center
│   ├── components/
│   │   └── dashboard.py
│   ├── styles/
│   │   └── theme.css
│   └── assets/
│       └── samples/
├── backend/
│   ├── config.py       # Centralized configuration & parameters
│   ├── detector.py     # YOLOv8 inference pipeline (logic unchanged)
│   └── services/
├── tests/
│   └── test_detector_regression.py
├── app.py              # Backward-compatible Streamlit entry point
├── weights/
│   └── best.pt         # Trained YOLOv8 model weights
└── requirements.txt    # Project dependencies
```
