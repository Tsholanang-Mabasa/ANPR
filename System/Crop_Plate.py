from Enhancer import *

def enhance_plate_crop(crop_bgr: np.ndarray, upscale=2) -> np.ndarray:
    if crop_bgr.size == 0:
        return crop_bgr

    out = clahe_on_luminance(crop_bgr, clip=2.6, tile=(6,6))
    out = auto_gamma(out, target_mean=0.55)
    out = unsharp_mask(out, sigma=0.9, amount=1.25)
    if upscale and (crop_bgr.shape[0] > 0 and crop_bgr.shape[1] > 0):
        out = cv2.resize(out, (crop_bgr.shape[1]*upscale, crop_bgr.shape[0]*upscale), interpolation=cv2.INTER_CUBIC)
        out = unsharp_mask(out, sigma=0.8, amount=0.6)
    return out