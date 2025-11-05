from pathlib import Path
import csv
import cv2
import numpy as np
import Detect_Plate as dp
import GT2 as gt2
from TesseractOCR import _ocr_tess

IMAGE_PATH = r"/home/tsholanang/PycharmProjects/ANPR/images/Test6.png"

# Compare Images ide by side
def _side_by_side(a, b, gap=8):
    h = max(a.shape[0], b.shape[0])
    wa = a.shape[1]; wb = b.shape[1]
    ca = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    cb = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    pad = np.zeros((h, gap, 3), dtype=np.uint8)
    return np.hstack([ca, pad, cb])

def run_pipeline():
    # ---------- load original ----------
    orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    stem = Path(IMAGE_PATH).stem
    OUT = Path("results") / stem
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------- "normal preprocessing" then build GT2 on it ----------
    pre = gt2.clahe_on_luma(orig)
    mapped = gt2.run_gt2_map_from_pre_to_orig(pre, orig, out_dir=OUT / "gt2_map", stem=stem)
    processed_bgr = cv2.imread(mapped["paths"]["mapped_dark"], cv2.IMREAD_COLOR)
    if processed_bgr is None:
        raise RuntimeError("Failed to create processed-mapped image.")

    # Save the processed image for YOLO input
    processed_path = str(OUT / f"{stem}_processed_mapped.png")
    cv2.imwrite(processed_path, processed_bgr)

    # ---------- detect on processed ----------
    R_p, P_p, crops_p, ann_p_path, boxes_p = dp.detect_plates(
        image_path=processed_path,
        model_path=dp.MODEL_PATH,
        confidence=getattr(dp, "CONFIDENCE", 0.25),
        annotate_conf=True,
    )

    # ---------- detect on original ----------
    R_o, P_o, crops_o, ann_o_path, boxes_o = dp.detect_plates(
        image_path=IMAGE_PATH,
        model_path=dp.MODEL_PATH,
        confidence=getattr(dp, "CONFIDENCE", 0.25),
        annotate_conf=True,
    )

    # ---------- side-by-side comparison image ----------
    ann_o = cv2.imread(str(ann_o_path), cv2.IMREAD_COLOR) if ann_o_path else orig.copy()
    ann_p = cv2.imread(str(ann_p_path), cv2.IMREAD_COLOR) if ann_p_path else processed_bgr.copy()

    cv2.putText(ann_o, "Original", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(ann_p, "Processed (GT2 edges mapped to original)", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

    comp = _side_by_side(ann_o, ann_p, gap=10)
    comp_path = OUT / "compare_original_vs_processed.jpg"
    cv2.imwrite(str(comp_path), comp)

    # ---------- OCR both sets ----------
    (OUT / "ocr_original").mkdir(exist_ok=True, parents=True)
    (OUT / "ocr_processed").mkdir(exist_ok=True, parents=True)

    rows = []

    def _do_ocr(crop_paths, tag):
        for cp in crop_paths:
            img = cv2.imread(str(cp), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read crop: {cp}")
                continue
            o = _ocr_tess(img)
            txt, confv = o["text"], o["conf"]
            dbg = img.copy()
            label = f"{txt} ({confv:.2f})" if txt else f"'' ({confv:.2f})"
            cv2.putText(dbg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            if tag == "original":
                dbg_path = OUT / "ocr_original" / (Path(cp).stem + "_ocr.jpg")
            else:
                dbg_path = OUT / "ocr_processed" / (Path(cp).stem + "_ocr.jpg")
            cv2.imwrite(str(dbg_path), dbg)

            rows.append({
                "variant": tag,
                "plate_crop": str(cp),
                "ocr_text": txt,
                "ocr_confidence": f"{confv:.4f}",
                "ocr_debug_image": str(dbg_path),
            })

    _do_ocr(crops_o, "original")
    _do_ocr(crops_p, "processed")

    # ---------- CSV summary ----------
    csv_path = OUT / "ocr_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "plate_crop", "ocr_text", "ocr_confidence", "ocr_debug_image"
        ])
        w.writeheader(); w.writerows(rows)

    print(f"\nDetection (original) annotated: {ann_o_path}")
    print(f"Detection (processed) annotated: {ann_p_path}")
    print(f"Side-by-side comparison: {comp_path}")
    print(f"OCR comparison rows: {len(rows)}  -> {csv_path}")

if __name__ == "__main__":
    run_pipeline()
