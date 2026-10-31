"""Drawing helpers; line width and font scale with image size."""
import cv2
import numpy as np

COLORS = {            # BGR
    "logo": (60, 200, 60),
    "face": (255, 140, 20),
    "object": (70, 70, 240),
}


def _thick(image):
    return max(2, int(round(max(image.shape[:2]) / 700)))


def _font(image):
    return max(0.5, max(image.shape[:2]) / 2000)


def draw_label(image, text, x, y, color):
    fs = _font(image)
    th = max(1, _thick(image) - 1)
    (tw, text_h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    y0 = max(text_h + base + 4, int(y))
    cv2.rectangle(image, (int(x), y0 - text_h - base - 4), (int(x) + tw + 8, y0), color, -1)
    cv2.putText(image, text, (int(x) + 4, y0 - base - 1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), th, cv2.LINE_AA)


def draw_box(image, box, label, kind):
    color = COLORS[kind]
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, _thick(image))
    if label:
        draw_label(image, label, x1, y1, color)


def draw_polygon(image, polygon, label, kind="logo"):
    color = COLORS[kind]
    pts = np.int32(polygon).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, _thick(image) + 1, cv2.LINE_AA)
    if label:
        draw_label(image, label, int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min()), color)
