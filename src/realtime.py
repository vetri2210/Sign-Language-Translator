import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from albumentations.pytorch import ToTensorV2
A.ToTensorV2 = ToTensorV2
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
import sys
import time


# =========================
# Initialize logger
# =========================
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# =========================
# Image transforms
# =========================
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()
])

# =========================
# Load model
# =========================
model = DETR(num_classes=3)
model.eval()
model.load_pretrained('checkpoints/50_model.pt')
CLASSES = get_classes()
COLORS = get_colors()

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logger.error("❌ Could not open camera. Please check your webcam connection.")
    sys.exit(1)

# =========================
# Check for GUI support
# =========================
HAS_GUI = hasattr(cv2, "imshow") and callable(cv2.imshow)
if not HAS_GUI:
    logger.warning("⚠️ OpenCV installed without GUI support — frames will be saved instead of displayed.")

# =========================
# Initialize performance tracking
# =========================
frame_count = 0
fps_start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        # Inference
        inference_start = time.time()
        transformed = transforms(image=frame)
        result = model(torch.unsqueeze(transformed['image'], dim=0))
        inference_time = (time.time() - inference_start) * 1000  # ms

        # Prediction postprocessing
        probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
        max_probs, max_classes = probabilities.max(-1)
        keep_mask = max_probs > 0.8
        batch_indices, query_indices = torch.where(keep_mask)

        bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (frame.shape[1], frame.shape[0]))
        classes = max_classes[batch_indices, query_indices]
        probas = max_probs[batch_indices, query_indices]

        detections = []
        for bclass, bprob, bbox in zip(classes, probas, bboxes):
            bclass_idx = int(bclass.detach().cpu().numpy())
            bprob_val = float(bprob.detach().cpu().numpy())
            x1, y1, x2, y2 = bbox.detach().cpu().numpy()

            detections.append({
                'class': CLASSES[bclass_idx],
                'confidence': bprob_val,
                'bbox': [x1, y1, x2, y2]
            })

            color = COLORS[bclass_idx]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            label = f"{CLASSES[bclass_idx]}: {bprob_val:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1) - 40), (int(x1) + 250, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1) + 10, int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # FPS calculation
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start_time
            fps = 30 / elapsed
            if detections:
                detection_handler.log_detections(detections, frame_id=frame_count)
            detection_handler.log_inference_time(inference_time, fps)
            fps_start_time = time.time()

        # =========================
        # DISPLAY or SAVE
        # =========================
        if HAS_GUI:
            cv2.imshow("SignDETR Real-Time Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.realtime("Stopping real-time detection...")
                break
        else:
            # fallback — no GUI
            cv2.imwrite("last_frame.jpg", frame)
            print(f"Frame saved as last_frame.jpg | Shape: {frame.shape}")

        time.sleep(0.01)  # prevent CPU overload

except KeyboardInterrupt:
    logger.realtime("Manually stopped via KeyboardInterrupt")

finally:
    cap.release()
    if HAS_GUI:
        cv2.destroyAllWindows()
    logger.realtime("Camera released and program exited cleanly.")
