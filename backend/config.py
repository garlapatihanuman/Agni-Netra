import os

# Model & Inference Defaults
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
DEVICE = os.getenv("DEVICE", "cpu").lower()
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "50"))

# Alert Defaults
ENABLE_AUDIO_ALARM = os.getenv("ENABLE_AUDIO_ALARM", "True").lower() in ("true", "1", "yes")
