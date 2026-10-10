from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from backend.config import (
    CONFIDENCE_THRESHOLD,
    DEVICE,
    ENABLE_AUDIO_ALARM,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MODEL_PATH,
)
from backend.detector import detect_objects, get_model_names, load_model
from frontend.components.dashboard import (
    render_3d_hologram_core,
    render_hazard_banner,
    render_telemetry_metrics,
)

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgniNetra | Tactical Fire & Flame Command",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Futuristic Cyberpunk & Glassmorphism UI Styling
# ---------------------------------------------------------
CUSTOM_CSS = (Path(__file__).resolve().parent / "styles" / "theme.css").read_text(encoding="utf-8")

st.markdown(f"<style>\n{CUSTOM_CSS}\n</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Backend Model Loader (Unchanged)
# ---------------------------------------------------------
@st.cache_resource
def get_loaded_model(model_path: str, device: str):
    """Cache loaded YOLO model in memory."""
    return load_model(model_path=model_path, device=device)


# ---------------------------------------------------------
# Tactical Navigation Bar & Hero Section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="tactical-navbar">
        <div class="brand-wrapper">
            <div class="brand-icon-box">AN</div>
            <div>
                <h1 class="brand-title">AGNINETRA</h1>
                <div class="brand-subtitle">Tactical Live Flame, Fire & Smoke Detection Suite • v2.4 Neural</div>
            </div>
        </div>
        <div class="nav-telemetry">
            <div class="status-pill">
                <span class="pulse-dot"></span>
                <span>SYSTEM ONLINE</span>
            </div>
            <div class="tech-tag">YOLOv8 NEURAL CORE</div>
            <div class="tech-tag">CHROMATIC FILTRATION</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-family: 'Orbitron', sans-serif; font-size: 1.15rem; font-weight: 700; color: #ff8c00; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            SYSTEM PARAMETERS
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = st.text_input("Model Weights Path", value=MODEL_PATH, help="Path to trained YOLOv8 .pt weights file")
    device_opt = DEVICE

    st.markdown("---")
    st.markdown('<div style="font-weight: 700; color: #b0c4de; font-family: \'Orbitron\', sans-serif; font-size: 0.88rem; margin-bottom: 8px;">DETECTION SENSITIVITY</div>', unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Higher values eliminate false alarms; lower values capture fainter flames and smoke.",
    )
    iou_threshold = st.slider(
        "IoU Overlap Threshold",
        min_value=0.10,
        max_value=0.90,
        value=IOU_THRESHOLD,
        step=0.05,
        help="Intersection-over-Union threshold for Non-Maximum Suppression.",
    )
    image_size = st.select_slider(
        "Inference Resolution",
        options=[320, 480, 640, 800, 1024],
        value=IMAGE_SIZE,
        help="640 provides an optimal balance between FPS and flame detection precision.",
    )
    max_detections = st.slider("Max Detections / Frame", min_value=5, max_value=100, value=MAX_DETECTIONS, step=5)

    st.markdown("---")
    st.markdown('<div style="font-weight: 700; color: #b0c4de; font-family: \'Orbitron\', sans-serif; font-size: 0.88rem; margin-bottom: 8px;">NEURAL INTEGRITY</div>', unsafe_allow_html=True)
    st.caption("Exclusively powered by trained weights (best.pt). Neural pattern recognition prevents false alarms on orange shirts, walls, or ambient lights while accurately locking onto live flame, fire, and smoke.")
    enable_audio = st.toggle(
        "Web Audio Alarm Siren",
        value=ENABLE_AUDIO_ALARM,
        help="Emits an audible emergency warning siren when live flame or fire is detected.",
    )

    st.markdown("---")
    if st.button("Reload Model Pipeline", use_container_width=True):
        st.cache_resource.clear()
        st.success(f"Cache cleared. Rebuilding pipeline from {model_path}")

# ---------------------------------------------------------
# Load Backend Model (Unchanged)
# ---------------------------------------------------------
try:
    model = get_loaded_model(model_path, device_opt)
    model_names = get_model_names(model)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# ---------------------------------------------------------
# Main Navigation Tabs
# (One-Shot / Image Processing Preset Demo completely removed)
# ---------------------------------------------------------
tab_live_cam, tab_snapshot_cam, tab_image, tab_video = st.tabs(
    [
        "Live Webcam Stream",
        "Instant Camera Snapshot",
        "Image Inspection & Tactical Upload",
        "Video File Analysis",
    ]
)

# ---------------------------------------------------------
# TAB 1: REAL-TIME CONTINUOUS LIVE WEBCAM STREAM
# ---------------------------------------------------------
def open_camera_with_fallback(camera_idx):
    camera_candidates = [camera_idx] + [
        index for index in (0, 1, 2) if index != camera_idx
    ]
    for index in camera_candidates:
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            return capture, index
        capture.release()
    return None, None


with tab_live_cam:
    st.markdown(
        """
        <div class="glass-panel glass-panel-accent">
            <div class="panel-header-title">
                REAL-TIME CONTINUOUS WEBCAM SURVEILLANCE
            </div>
            <p class="panel-desc">
                Streams live high-framerate feed directly from your camera, continuously executing the trained YOLOv8 neural network.
                Engineered for rapid detection of live lighter flames, candles, matches, open fire outbreaks, and smoke plumes with tactical HUD telemetry.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_cam_ctrl1, col_cam_ctrl2, col_cam_ctrl3 = st.columns([2, 2, 2])
    with col_cam_ctrl1:
        camera_idx = st.selectbox(
            "Camera Device Index",
            options=[0, 1, 2],
            index=0,
            help="0 is usually the built-in camera. If unavailable, the app tries the other listed camera indexes.",
        )
    with col_cam_ctrl2:
        stream_resolution = st.selectbox("Stream Resolution", ["640x480 (Fastest)", "1280x720 (High-Res)"], index=0)
    with col_cam_ctrl3:
        pass

    if "live_webcam_running" not in st.session_state:
        st.session_state.live_webcam_running = False

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("START LIVE WEBCAM STREAM", use_container_width=True, type="primary"):
            st.session_state.live_webcam_running = True
    with btn_col2:
        if st.button("STOP LIVE WEBCAM STREAM", use_container_width=True):
            st.session_state.live_webcam_running = False

    live_frame_placeholder = st.empty()
    live_banner_placeholder = st.empty()
    live_metrics_placeholder = st.empty()

    st.markdown("#### System Visualization")
    st.caption(
        "Animated system-state view only. Use the camera feed above for detected objects, "
        "bounding boxes, class labels, and confidence scores."
    )
    live_visualization_placeholder = st.empty()
    current_hazard = st.session_state.get("last_hazard", "STANDBY")
    active_classes = st.session_state.get("last_classes", "")
    with live_visualization_placeholder.container():
        render_3d_hologram_core(current_hazard, active_classes)
    last_visual_signature = (current_hazard, active_classes)

    if st.session_state.live_webcam_running:
        cap, active_camera_idx = open_camera_with_fallback(camera_idx)

        if cap is None:
            st.error(
                "Could not open a camera at indexes 0, 1, or 2. Check that a camera is connected "
                "and allow camera access for the app running Streamlit. On macOS, check "
                "System Settings > Privacy & Security > Camera for Terminal or VS Code, then try again."
            )
            st.session_state.live_webcam_running = False
        else:
            if active_camera_idx != camera_idx:
                st.info(
                    f"Camera index {camera_idx} was unavailable; using camera index {active_camera_idx} instead."
                )

            w_req, h_req = (640, 480) if "640" in stream_resolution else (1280, 720)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_req)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_req)

            fps_timer = time.time()
            fps_counter = 0
            live_fps = 0.0

            while st.session_state.live_webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam stream disconnected.")
                    break

                # Inference via unchanged backend
                annotated_frame, stats = detect_objects(
                    frame=frame,
                    model=model,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    max_detections=max_detections,
                    device=device_opt,
                )

                # Keep the secondary visualization tied to detector results.
                st.session_state["last_hazard"] = stats["hazard_level"]
                st.session_state["last_classes"] = ", ".join(stats["class_counts"].keys())
                visual_signature = (
                    stats["hazard_level"],
                    ", ".join(stats["class_counts"].keys()),
                )
                if visual_signature != last_visual_signature:
                    with live_visualization_placeholder.container():
                        render_3d_hologram_core(*visual_signature)
                    last_visual_signature = visual_signature

                # FPS Calculation
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    live_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # Add Tactical HUD telemetry on video overlay
                cv2.putText(
                    annotated_frame,
                    f"AGNINETRA LIVE | FPS: {live_fps:.1f} | LATENCY: {stats['inference_time_ms']:.0f}ms | TARGETS: {stats['total_detections']}",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 240, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Keep detector output visually primary over supporting telemetry.
                live_frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                with live_banner_placeholder.container():
                    render_hazard_banner(stats, enable_audio=enable_audio)
                with live_metrics_placeholder.container():
                    render_telemetry_metrics(stats)

            cap.release()

# ---------------------------------------------------------
# TAB 2: INSTANT CAMERA SNAPSHOT
# ---------------------------------------------------------
with tab_snapshot_cam:
    st.markdown(
        """
        <div class="glass-panel glass-panel-accent">
            <div class="panel-header-title">
                INSTANT CAMERA SNAPSHOT INSPECTION
            </div>
            <p class="panel-desc">
                Capture high-resolution single frames via browser camera input. Ideal for fast spot-checks, low-bandwidth networks, or precise frame-by-frame verification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cam_snapshot = st.camera_input("Capture live photo for instant flame/smoke inspection", key="snapshot_input")
    if cam_snapshot is not None:
        try:
            pil_img = Image.open(cam_snapshot).convert("RGB")
            frame_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing captured frame with neural detector..."):
                annotated_frame, stats = detect_objects(
                    frame=frame_cv,
                    model=model,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    max_detections=max_detections,
                    device=device_opt,
                )

            # Update 3D Visualizer state
            st.session_state["last_hazard"] = stats["hazard_level"]
            st.session_state["last_classes"] = ", ".join(stats["class_counts"].keys())

            render_hazard_banner(stats, enable_audio=enable_audio)
            render_telemetry_metrics(stats)

            col_s1, col_s2 = st.columns([3, 2])
            with col_s1:
                st.image(annotated_frame, channels="BGR", caption="Tactical Detection Output", use_container_width=True)
            with col_s2:
                st.markdown("#### Detection Registry")
                if stats["detections"]:
                    det_table = [
                        {
                            "Target": d["class_name"],
                            "Confidence": f"{d['confidence']:.1%}",
                            "BBox [x1, y1, x2, y2]": f"{d['bbox']}",
                            "Engine": d.get("source", "yolo_best_pt"),
                        }
                        for d in stats["detections"]
                    ]
                    st.dataframe(det_table, use_container_width=True)
                else:
                    st.info("No fire, live flame, or smoke signatures detected.")
        except Exception as err:
            st.error(f"Error processing camera snapshot: {err}")

# ---------------------------------------------------------
# TAB 3: IMAGE INSPECTION & TACTICAL UPLOAD
# (Clean, Dedicated Image Upload Interface — One-Shot / Preset Demo Completely Removed)
# ---------------------------------------------------------
with tab_image:
    st.markdown(
        """
        <div class="glass-panel glass-panel-accent">
            <div class="panel-header-title">
                HIGH-PRECISION IMAGE INSPECTION & UPLOAD
            </div>
            <p class="panel-desc">
                Upload custom aerial, structural, or handheld surveillance imagery for thorough neural inspection.
                The YOLOv8 neural network accurately isolates genuine combustion signatures while disregarding false triggers like orange clothing or warm illumination.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_img = st.file_uploader(
        "Upload Image for AI Combustion Inspection",
        type=["png", "jpg", "jpeg", "webp"],
        key="tactical_image_uploader",
        help="Upload PNG, JPG, JPEG, or WEBP imagery for tactical flame/fire/smoke inference",
    )

    if uploaded_img is not None:
        try:
            pil_file = Image.open(uploaded_img).convert("RGB")
            image_source = cv2.cvtColor(np.array(pil_file), cv2.COLOR_RGB2BGR)

            with st.spinner("Executing neural inspection pipeline..."):
                annotated_img, stats = detect_objects(
                    frame=image_source,
                    model=model,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    image_size=image_size,
                    max_detections=max_detections,
                    device=device_opt,
                )

            # Update 3D Visualizer State
            st.session_state["last_hazard"] = stats["hazard_level"]
            st.session_state["last_classes"] = ", ".join(stats["class_counts"].keys())

            render_hazard_banner(stats, enable_audio=enable_audio)
            render_telemetry_metrics(stats)

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("##### Raw Input Source")
                st.image(image_source, channels="BGR", caption="Original Source Image", use_container_width=True)
            with img_col2:
                st.markdown("##### Tactical AI Output")
                st.image(annotated_img, channels="BGR", caption="AI Annotated Detection (HUD Reticles)", use_container_width=True)

            # Target breakdown registry & download
            st.markdown("#### Target Classification Registry")
            if stats["detections"]:
                det_data = [
                    {
                        "Target Class": d["class_name"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "Bounding Box [x1, y1, x2, y2]": f"{d['bbox']}",
                        "Engine": d.get("source", "yolo_best_pt"),
                    }
                    for d in stats["detections"]
                ]
                st.dataframe(det_data, use_container_width=True)

                # Download annotated image button
                ret_enc, buf_enc = cv2.imencode(".jpg", annotated_img)
                if ret_enc:
                    st.download_button(
                        label="Download Annotated Image",
                        data=buf_enc.tobytes(),
                        file_name="ignis_tactical_detection.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )
            else:
                st.info("Surveillance verification complete: No fire, live flame, or smoke targets identified.")
        except Exception as e:
            st.error(f"Error analyzing uploaded image: {e}")
    else:
        st.markdown(
            """
            <div class="empty-state-box">
                <div style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                    AWAITING SURVEILLANCE IMAGERY
                </div>
                <div style="color: #9db2ce; font-size: 0.88rem; max-width: 520px; margin: 0 auto; line-height: 1.5;">
                    Drag and drop or browse to upload an image above. Supported formats include <span style="color: #00f0ff;">PNG, JPG, JPEG, and WEBP</span>. Real-time predictions, bounding reticles, and hazard telemetry will populate automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# TAB 4: VIDEO FILE ANALYSIS
# ---------------------------------------------------------
with tab_video:
    st.markdown(
        """
        <div class="glass-panel glass-panel-accent">
            <div class="panel-header-title">
                VIDEO STREAM FILE ANALYSIS
            </div>
            <p class="panel-desc">
                Upload recorded video clips for continuous multi-frame hazard tracking, frame-level statistics, and incident analytics breakdown.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_vid = st.file_uploader(
        "Upload Video File",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader",
    )

    if uploaded_vid is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1] or ".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            v_cap = cv2.VideoCapture(tmp_path)
            total_vid_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_vid = v_cap.get(cv2.CAP_PROP_FPS) or 30.0

            st.caption(f"Video Loaded: {total_vid_frames} frames @ {fps_vid:.1f} FPS")

            progress_bar = st.progress(0, text="Processing video frames...")
            video_frame_placeholder = st.empty()
            video_metrics_placeholder = st.empty()

            processed_count = 0
            hazard_frames = 0
            peak_conf = 0.0
            class_summary: Counter[str] = Counter()

            while v_cap.isOpened():
                ret, frame = v_cap.read()
                if not ret:
                    break

                processed_count += 1
                # Sample every 2nd frame for responsive playback speed
                if processed_count % 2 == 0 or processed_count == total_vid_frames:
                    annotated_v, v_stats = detect_objects(
                        frame=frame,
                        model=model,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        image_size=image_size,
                        max_detections=max_detections,
                        device=device_opt,
                    )

                    if v_stats["total_detections"] > 0:
                        hazard_frames += 1
                        peak_conf = max(peak_conf, v_stats["highest_confidence"])
                        for c_name, count in v_stats["class_counts"].items():
                            class_summary[c_name] += count

                    video_frame_placeholder.image(annotated_v, channels="BGR", use_container_width=True)

                progress_bar.progress(
                    min(1.0, processed_count / float(total_vid_frames)),
                    text=f"Processing frame {processed_count}/{total_vid_frames}...",
                )

            v_cap.release()
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            progress_bar.empty()
            st.success(f"Video analysis complete! Processed {processed_count} frames.")

            sum_c1, sum_c2, sum_c3 = st.columns(3)
            sum_c1.metric("Hazard Frames Detected", f"{hazard_frames} / {processed_count}")
            sum_c2.metric("Peak Target Confidence", f"{peak_conf:.1%}")
            sum_c3.metric("Total Incident Flags", sum(class_summary.values()))

            if class_summary:
                st.write("Incident Classification Breakdown:")
                st.bar_chart(dict(class_summary))
        except Exception as e:
            st.error(f"Error during video analysis: {e}")
