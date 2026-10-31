"""
Face detection.

Preferred: OpenCV YuNet (accurate, handles small/rotated faces). Its ~230 KB ONNX model is
downloaded once into ./models. If it cannot be downloaded (offline), the app automatically
falls back to bundled Haar cascades, which need no download.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .logo_search import box_iou_xyxy

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
YUNET_FILE = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _ensure_yunet():
    if YUNET_FILE.exists() and YUNET_FILE.stat().st_size > 100_000:
        return True
    try:
        MODEL_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, YUNET_FILE)
        return YUNET_FILE.stat().st_size > 100_000
    except Exception:
        if YUNET_FILE.exists():
            YUNET_FILE.unlink(missing_ok=True)
        return False


def _contained(a, b):
    """Fraction of box `a` that lies inside box `b` (suppresses nested false positives)."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _size_category(box, img_shape):
    h, w = img_shape[:2]
    rel = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
    if rel >= 0.02:
        return "Large"
    if rel >= 0.002:
        return "Medium"
    return "Small"


def _position(box, img_shape):
    h, w = img_shape[:2]
    cx = (box[0] + box[2]) / 2 / w
    return "Left" if cx < 0.33 else ("Right" if cx > 0.66 else "Center")


def _yunet_detect(img, min_conf):
    h, w = img.shape[:2]
    scale = min(1.0, 1920.0 / max(h, w))
    work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    det = cv2.FaceDetectorYN.create(str(YUNET_FILE), "", (work.shape[1], work.shape[0]),
                                    min_conf, 0.3, 5000)
    _, faces = det.detect(work)
    out = []
    for f in (faces if faces is not None else []):
        x, y, bw, bh = f[:4] / scale
        out.append({"box": [float(x), float(y), float(x + bw), float(y + bh)],
                    "confidence": round(float(f[14]), 4), "pose": "Frontal/Angled"})
    return out


_CASCADES = {}


def _cascade(name):
    if name not in _CASCADES:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        if c.empty():
            raise RuntimeError(f"Could not load cascade {name}")
        _CASCADES[name] = c
    return _CASCADES[name]


def _tiles(h, w, tile=1100, overlap=220):
    """Overlapping tiles so faces are examined at native resolution in huge images."""
    def spans(n):
        if n <= tile:
            return [(0, n)]
        step, out, a = tile - overlap, [], 0
        while True:
            b = min(a + tile, n)
            out.append((a, b))
            if b >= n:
                return out
            a += step
    return [(x0, y0, x1, y1) for (y0, y1) in spans(h) for (x0, x1) in spans(w)]


def _haar_detect(img, min_neighbors=5, include_profile=False):
    """Frontal faces from two cascades (+ optional profile); a candidate must be confirmed
    by the second cascade or by an eye detection, which removes most false positives."""
    h, w = img.shape[:2]
    work, scale = img, 1.0
    if max(h, w) > 6000:                       # keep runtime bounded on gigantic images
        scale = 6000.0 / max(h, w)
        work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray_full = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
    gh, gw = gray_full.shape[:2]
    c_def = _cascade("haarcascade_frontalface_default.xml")
    c_alt = _cascade("haarcascade_frontalface_alt2.xml")
    c_eye = _cascade("haarcascade_eye.xml")
    c_prof = _cascade("haarcascade_profileface.xml") if include_profile else None
    min_face = 24

    def run(c, g, nn):
        r = c.detectMultiScale(g, scaleFactor=1.08, minNeighbors=nn, minSize=(min_face, min_face))
        return [tuple(map(int, b)) for b in r] if len(r) else []

    def to_box(x, y, bw, bh, ox, oy):
        return [(x + ox) / scale, (y + oy) / scale, (x + bw + ox) / scale, (y + bh + oy) / scale]

    d_boxes, a_boxes, p_boxes = [], [], []
    for (x0, y0, x1, y1) in _tiles(gh, gw):
        g = gray_full[y0:y1, x0:x1]
        d_boxes += [to_box(*b, x0, y0) for b in run(c_def, g, min_neighbors)]
        a_boxes += [to_box(*b, x0, y0) for b in run(c_alt, g, min_neighbors)]
        if c_prof is not None:
            tw = g.shape[1]
            for flip in (False, True):
                gg = cv2.flip(g, 1) if flip else g
                for (x, y, bw, bh) in run(c_prof, gg, min_neighbors + 1):
                    if flip:
                        x = tw - x - bw
                    p_boxes.append(to_box(x, y, bw, bh, x0, y0))

    def has_eyes(box):
        x1, y1, x2, y2 = [int(v * scale) for v in box]
        roi = gray_full[max(0, y1):int(y1 + 0.62 * (y2 - y1)), max(0, x1):x2]
        if roi.size == 0 or min(roi.shape[:2]) < 20:
            return False
        eyes = c_eye.detectMultiScale(roi, 1.1, 4, minSize=(max(6, (x2 - x1) // 10),) * 2)
        return len(eyes) >= 1

    out = []
    for b in d_boxes + a_boxes:
        agree = any(box_iou_xyxy(b, o) > 0.4 for o in (a_boxes if b in d_boxes else d_boxes))
        if agree:
            out.append({"box": b, "confidence": 0.9, "pose": "Frontal"})
        elif has_eyes(b):
            out.append({"box": b, "confidence": 0.7, "pose": "Frontal"})
    for b in p_boxes:
        if all(box_iou_xyxy(b, f["box"]) < 0.3 for f in out):
            out.append({"box": b, "confidence": 0.55, "pose": "Profile"})
    return out


def detect_faces(img_bgr, min_confidence=0.7, prefer_dnn=True, include_profile=False):
    """Return (faces, info). Each face: box, confidence, pose, size, position."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Image is empty.")
    method, faces = "Haar cascades", []
    if prefer_dnn and hasattr(cv2, "FaceDetectorYN") and _ensure_yunet():
        try:
            faces = _yunet_detect(img_bgr, min_confidence)
            method = "YuNet (DNN)"
        except Exception:
            faces, method = [], "Haar cascades"
    if method == "Haar cascades":
        faces = _haar_detect(img_bgr, include_profile=include_profile)

    faces.sort(key=lambda f: f["confidence"], reverse=True)
    kept = []
    for f in faces:
        if all(box_iou_xyxy(f["box"], k["box"]) < 0.35 and _contained(f["box"], k["box"]) < 0.6
               for k in kept):
            kept.append(f)
    kept.sort(key=lambda f: (f["box"][1] // 80, f["box"][0]))   # reading order
    for i, f in enumerate(kept, 1):
        f["id"] = i
        f["size"] = _size_category(f["box"], img_bgr.shape)
        f["position"] = _position(f["box"], img_bgr.shape)
    return kept, {"method": method, "count": len(kept)}
