import cv2
import numpy as np

def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    m = (mb + mg + mr) / 3.0 + 1e-6
    b *= m / (mb + 1e-6); g *= m / (mg + 1e-6); r *= m / (mr + 1e-6)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_on_luminance(bgr: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def auto_gamma(bgr: np.ndarray, target_mean=0.5) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean()) + 1e-6
    gamma = np.clip(np.log(target_mean) / np.log(mean), 0.6, 1.6)
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bgr, table)

def unsharp_mask(bgr: np.ndarray, blur_ks=0, sigma=1.2, amount=1.2) -> np.ndarray:
    if blur_ks == 0:
        blur_ks = int(max(3, 2 * int(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(bgr, (blur_ks, blur_ks), sigma)
    sharp = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)
