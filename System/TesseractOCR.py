from pathlib import Path
import cv2
import numpy as np
import pytesseract
import re

_PLATE_RE = re.compile(r"[A-Z0-9-]+")

def _norm_plate(text: str) -> str:
    text = (text or "").upper()
    text = re.sub(r"[^A-Z0-9-]", "", text)
    return text.strip()

def _ocr_tess(bgr, config="--oem 3 --psm 7 -l eng"):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    texts, confs = [], []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if txt and str(txt).strip():
            texts.append(str(txt).strip())
            try:
                confs.append(float(conf))
            except:
                pass
    joined = _norm_plate("".join(texts))
    if not joined:
        return {"text": "", "conf": 0.0}
    confv = (np.mean(confs) / 100.0) if confs else 0.5
    return {"text": joined, "conf": float(confv)}
