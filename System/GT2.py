import cv2, numpy as np
from pathlib import Path

SCALE_FOR_GT2    = 1.0
Z_LEVELS         = [0.2, 0.4, 0.6, 0.8, 1.0]
GT2_GAMMA        = 0.7
GT2_SHARP        = 0.35
CLOSE_KERNEL     = (3, 3)
MASK_POWER       = 1.45
SHARP_SIGMA      = 1.25
SHARP_AMOUNT     = 1.05
EDGE_DARK_AMOUNT = 0.30
CLAHE_CLIP       = 2.0
CLAHE_TILE       = (8, 8)
SKEW_MAX_DEG     = 25.0

SETS = {
    "g": {
        "Low":    {"mu_bounds": (0.00, 0.15), "sigma_bounds": (0.10, 0.20)},
        "Medium": {"mu_bounds": (0.42, 0.58), "sigma_bounds": (0.15, 0.25)},
        "High":   {"mu_bounds": (0.70, 0.95), "sigma_bounds": (0.08, 0.15)},
    },
    "v": {
        "Low":  {"mu_bounds": (0.00, 0.25), "sigma_bounds": (0.12, 0.25)},
        "High": {"mu_bounds": (0.60, 0.95), "sigma_bounds": (0.10, 0.20)},
    },
}
CONSEQUENTS = {"Strong": 1.00, "Medium": 0.70, "Weak": 0.15}

# ------------- utils -------------
def _normalize01(a):
    a = a.astype(np.float32); mn, mx = float(a.min()), float(a.max())
    return np.zeros_like(a) if mx <= mn + 1e-12 else (a - mn) / (mx - mn)

def _gauss(x, mu, s):
    s = max(1e-6, float(s))
    z = -0.5 * ((x.astype(np.float32) - np.float32(mu)) / np.float32(s)) ** 2
    return np.exp(z, dtype=np.float32)

def grad_mag(gray):
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return _normalize01(cv2.magnitude(gx, gy))

def local_variance(gray, k=5):
    f = gray.astype(np.float32); mean = cv2.blur(f, (k, k)); mean2 = cv2.blur(f*f, (k, k))
    return _normalize01(mean2 - mean*mean)

def rotate_bound(img, angle_deg):
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW = int((h * sin) + (w * cos)); nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX; M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _zslices(x, mu_bounds, sigma_bounds, z_levels):
    muL, muU = mu_bounds; sL, sU = sigma_bounds
    return [_gauss(x, muL + (muU - muL) * z, sL + (sU - sL) * z) for z in z_levels]

# ------------- GT2 map -------------
def gt2_edge_score(g, v, z_levels=Z_LEVELS, sets=SETS, consequents=CONSEQUENTS):
    g_low = _zslices(g, **sets["g"]["Low"], z_levels=z_levels)
    g_med = _zslices(g, **sets["g"]["Medium"], z_levels=z_levels)
    g_high = _zslices(g, **sets["g"]["High"], z_levels=z_levels)
    v_low = _zslices(v, **sets["v"]["Low"], z_levels=z_levels)
    v_high = _zslices(v, **sets["v"]["High"], z_levels=z_levels)

    zsum = float(sum(z_levels))
    acc = np.zeros_like(g, dtype=np.float32)
    cS, cM, cW = consequents["Strong"], consequents["Medium"], consequents["Weak"]

    for i, z in enumerate(z_levels):
        gl, gm, gh = g_low[i], g_med[i], g_high[i]
        vl, vh = v_low[i], v_high[i]
        r1 = np.minimum(gh, vh)  # Strong
        r2 = np.minimum(gm, vh)  # Medium→Strong
        r3 = np.maximum(gl, vl)  # Weak
        r4 = np.minimum(gh, vl)  # Medium
        denom = r1 + r2 + r4 + r3 + 1e-8
        yz = (r1 * cS + r2 * cS + r4 * cM + r3 * cW) / denom
        acc += z * yz

    y = acc / (zsum + 1e-8)
    y = np.power(y, GT2_GAMMA)
    if GT2_SHARP > 0:
        lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
        y = np.clip(y + GT2_SHARP * lap, 0, 1)
    return _normalize01(y)

def gt2_edge_maps(gray):
    g = cv2.medianBlur(gray, 3); g = cv2.GaussianBlur(g, (3, 3), 0)
    G = grad_mag(g); V = local_variance(g, 5)
    y = gt2_edge_score(G, V)
    soft = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    soft = cv2.morphologyEx(soft, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL))
    _, hard = cv2.threshold(soft, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return soft, hard

def estimate_skew_from_mask(mask, max_deg=SKEW_MAX_DEG):
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=120)
    if lines is None: return 0.0
    angs = []
    for rt in lines[:200]:
        _, theta = rt[0]
        ang = (theta * 180.0 / np.pi) - 90.0
        if -max_deg <= ang <= max_deg: angs.append(ang)
    return float(np.median(angs)) if angs else 0.0

def unsharp_on_luma(bgr, sigma=SHARP_SIGMA, amount=SHARP_AMOUNT):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    Ls = np.clip(cv2.addWeighted(L, 1 + amount, blur, -amount, 0), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([Ls, a, b]), cv2.COLOR_LAB2BGR)
    lap = cv2.Laplacian(out, cv2.CV_16S, ksize=3)
    lap = np.clip(lap, -8, 8).astype(np.int16)
    out = np.clip(out.astype(np.int16) - lap, 0, 255).astype(np.uint8)
    return out

def edge_aware_enhance_with_mask(bgr_base, soft_mask):
    m = cv2.GaussianBlur(soft_mask.astype(np.float32) / 255.0, (5, 5), 0)
    m = np.clip(m, 0, 1) ** MASK_POWER
    m3 = cv2.merge([m, m, m])

    sharp = unsharp_on_luma(bgr_base)
    enh = (bgr_base.astype(np.float32) * (1 - m3) + sharp.astype(np.float32) * m3).astype(np.uint8)

    lab = cv2.cvtColor(enh, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    grad = grad_mag(cv2.cvtColor(bgr_base, cv2.COLOR_BGR2GRAY))
    burn = (m * grad) * (EDGE_DARK_AMOUNT * 255.0)
    Ld = np.clip(L.astype(np.float32) - burn, 0, 255).astype(np.uint8)
    dark = cv2.cvtColor(cv2.merge([Ld, a, b]), cv2.COLOR_LAB2BGR)
    return enh, dark


def shade_correct(gray, k=41):
    bg = cv2.blur(gray, (k, k)).astype(np.float32); bg = np.clip(bg, 1, 255)
    flat = (gray.astype(np.float32) / bg) * 128.0
    return np.clip(flat, 0, 255).astype(np.uint8)

def clahe_on_luma(bgr, clip=CLAHE_CLIP, tile=CLAHE_TILE):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    Lc = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([Lc,a,b]), cv2.COLOR_LAB2BGR)

def run_gt2_on_bgr(bgr0, out_dir: Path, stem: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    H, W = bgr0.shape[:2]; gray0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)

    if SCALE_FOR_GT2 != 1.0:
        small = cv2.resize(gray0, (int(W*SCALE_FOR_GT2), int(H*SCALE_FOR_GT2)), interpolation=cv2.INTER_AREA)
        soft_s, _ = gt2_edge_maps(small)
        soft = cv2.resize(soft_s, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        soft, _ = gt2_edge_maps(gray0)

    angle = estimate_skew_from_mask(soft, SKEW_MAX_DEG)
    if abs(angle) > 0.3:
        bgr1  = rotate_bound(bgr0, -angle)
        soft1 = rotate_bound(soft,  -angle)
    else:
        bgr1, soft1 = bgr0, soft

    enh_bgr, dark_bgr = edge_aware_enhance_with_mask(bgr1, soft1)

    glare_fixed = clahe_on_luma(dark_bgr, clip=CLAHE_CLIP, tile=CLAHE_TILE)
    gray_enh = cv2.cvtColor(glare_fixed, cv2.COLOR_BGR2GRAY)
    flat = shade_correct(gray_enh, k=41)

    paths = {
        "deskewed":      str(out_dir / f"{stem}_deskewed.png"),
        "soft":          str(out_dir / f"{stem}_soft.png"),
        "edge_darkened": str(out_dir / f"{stem}_edge_darkened.png"),
        "glare_fixed":   str(out_dir / f"{stem}_glare_fixed.png"),
        "flat":          str(out_dir / f"{stem}_flat.png"),
    }
    cv2.imwrite(paths["deskewed"], bgr1)
    cv2.imwrite(paths["soft"], soft1)
    cv2.imwrite(paths["edge_darkened"], dark_bgr)
    cv2.imwrite(paths["glare_fixed"], glare_fixed)
    cv2.imwrite(paths["flat"], flat)

    return {"angle": float(angle), "paths": paths, "soft_mask": soft1}

def run_gt2_map_from_pre_to_orig(pre_bgr, orig_bgr, out_dir: Path, stem: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    H, W = orig_bgr.shape[:2]

    gray_pre = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2GRAY)

    if SCALE_FOR_GT2 != 1.0:
        small = cv2.resize(gray_pre, (int(W*SCALE_FOR_GT2), int(H*SCALE_FOR_GT2)), interpolation=cv2.INTER_AREA)
        soft_s, _ = gt2_edge_maps(small)
        soft = cv2.resize(soft_s, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        soft, _ = gt2_edge_maps(gray_pre)

    angle = estimate_skew_from_mask(soft, SKEW_MAX_DEG)
    if abs(angle) > 0.3:
        orig1 = rotate_bound(orig_bgr, -angle)
        soft1 = rotate_bound(soft, -angle)
    else:
        orig1, soft1 = orig_bgr, soft

    enh_map, dark_map = edge_aware_enhance_with_mask(orig1, soft1)

    paths = {
        "mapped_enh":  str(out_dir / f"{stem}_mapped_enh.png"),
        "mapped_dark": str(out_dir / f"{stem}_mapped_dark.png"),
        "soft_from_pre": str(out_dir / f"{stem}_soft_from_pre.png"),
        "orig_deskewed": str(out_dir / f"{stem}_orig_deskewed.png"),
    }
    cv2.imwrite(paths["mapped_enh"],  enh_map)
    cv2.imwrite(paths["mapped_dark"], dark_map)
    cv2.imwrite(paths["soft_from_pre"], soft1)
    cv2.imwrite(paths["orig_deskewed"], orig1)

    return {"angle": float(angle), "paths": paths, "soft_mask": soft1}

def sauvola(gray, win=31, k=0.2, R=128):
    f = gray.astype(np.float32)
    mean  = cv2.boxFilter(f, -1, (win, win), normalize=True)
    mean2 = cv2.boxFilter(f*f, -1, (win, win), normalize=True)
    std = np.sqrt(np.maximum(mean2 - mean*mean, 0))
    T = mean * (1 + k * (std / R - 1))
    return (f > T).astype(np.uint8) * 255

def hybrid_binarize(gray):
    b1 = sauvola(gray, win=31, k=0.20, R=128)
    b2 = sauvola(gray, win=51, k=0.18, R=128)
    _, bo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.max(cv2.max(b1, b2), bo)
