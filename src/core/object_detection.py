"""
Object detection + categorization with Ultralytics YOLO (COCO classes).
Detected classes are grouped into human-friendly super-categories.
The app keeps working (with a clear message) if ultralytics / weights are unavailable.
"""
_MODELS = {}

SUPER_CATEGORIES = {
    "People": ["person"],
    "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "Street objects": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "Bags & accessories": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "Sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "Kitchen & food": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                       "donut", "cake"],
    "Furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "Electronics": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven", "toaster", "sink", "refrigerator"],
    "Household": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}
_CLASS_TO_GROUP = {c: g for g, cs in SUPER_CATEGORIES.items() for c in cs}


def category_of(class_name):
    return _CLASS_TO_GROUP.get(class_name, "Other")


def _load(model_name):
    if model_name not in _MODELS:
        from ultralytics import YOLO
        _MODELS[model_name] = YOLO(model_name)
    return _MODELS[model_name]


def detect_objects(img_bgr, confidence=0.35, model_name="yolov8n.pt", imgsz=1280):
    """Return (objects, info). Never raises for missing YOLO; reports in info['error']."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Image is empty.")
    try:
        model = _load(model_name)
    except Exception as exc:  # ImportError, download failure, ...
        return [], {"model": model_name, "count": 0,
                    "error": f"YOLO unavailable ({type(exc).__name__}: {exc}). "
                             "Run `pip install ultralytics` and make sure the weights can download."}
    res = model.predict(source=img_bgr, conf=confidence, imgsz=imgsz, verbose=False)[0]
    objects = []
    if res.boxes is not None:
        for b in res.boxes:
            name = res.names.get(int(b.cls[0].item()), "object")
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            objects.append({"class": name, "category": category_of(name),
                            "confidence": round(float(b.conf[0].item()), 4),
                            "box": [x1, y1, x2, y2]})
    objects.sort(key=lambda o: (o["category"], -o["confidence"]))
    for i, o in enumerate(objects, 1):
        o["id"] = i
    return objects, {"model": model_name, "count": len(objects), "error": None}
