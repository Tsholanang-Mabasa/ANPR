from pathlib import Path
import cv2
from ultralytics import YOLO


MODEL_PATH = r"/home/tsholanang/PycharmProjects/ANPR/Model/best.pt"
CONFIDENCE = 0.25  # YOLO detection threshold

def _draw_label(img, x1, y1, label):
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y), (x1 + tw + 6, y + th + 6), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 3, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detect_plates(image_path: str,
                  model_path: str = MODEL_PATH,
                  confidence: float = CONFIDENCE,
                  annotate_conf: bool = True):

    img_path = Path(image_path)
    base = img_path.stem
    results_dir = Path("results") / base
    plates_dir = results_dir / "DetectedPlates"
    results_dir.mkdir(parents=True, exist_ok=True)
    plates_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(model_path)
    result = model(img[:, :, ::-1], conf=confidence, verbose=False)[0]  # BGR->RGB

    saved = []
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        print("No plates detected.")
        return results_dir, plates_dir, saved, None, boxes

    annotated = img.copy()
    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1] - 1, x2); y2 = min(img.shape[0] - 1, y2)
        conf = float(box.conf[0].cpu().item()) if hasattr(box, "conf") else 0.0

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if annotate_conf:
            _draw_label(annotated, x1, y1, f"{conf:.2f}")

        crop = img[y1:y2, x1:x2].copy()
        out_path = plates_dir / f"plate_{i}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved.append(out_path)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf})

    ann_path = results_dir / "full_detected_annotated.jpg"
    cv2.imwrite(str(ann_path), annotated)
    print(f"Saved {len(saved)} plate crops -> {plates_dir}")
    return results_dir, plates_dir, saved, ann_path, boxes
