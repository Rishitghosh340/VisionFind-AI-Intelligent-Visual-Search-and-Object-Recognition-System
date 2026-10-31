"""Build downloadable reports: JSON, CSV and a ZIP with annotated image + crops."""
import io
import json
import zipfile

import pandas as pd

from .image_io import crop, encode_png


def detections_frame(result):
    rows = []
    for l in result["logos"]:
        x1, y1, x2, y2 = l["box"]
        rows.append({"type": "logo", "id": l["id"], "label": "reference match", "category": "Logo",
                     "confidence": l["confidence"], "method": l["method"],
                     "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    for f in result["faces"]:
        x1, y1, x2, y2 = f["box"]
        rows.append({"type": "face", "id": f["id"], "label": "face", "category": f"{f['size']} / {f['position']}",
                     "confidence": f["confidence"], "method": result["diagnostics"].get("faces", {}).get("method", ""),
                     "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    for o in result["objects"]:
        x1, y1, x2, y2 = o["box"]
        rows.append({"type": "object", "id": o["id"], "label": o["class"], "category": o["category"],
                     "confidence": o["confidence"], "method": "YOLO",
                     "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
    return pd.DataFrame(rows)


def result_json(result):
    payload = {k: result[k] for k in ("summary", "timing_ms", "diagnostics", "logos", "faces", "objects")}
    return json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def build_zip(result, scene_bgr):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotated.png", encode_png(result["image"]))
        z.writestr("detections.csv", detections_frame(result).to_csv(index=False))
        z.writestr("report.json", result_json(result))
        for l in result["logos"]:
            c = crop(scene_bgr, l["box"], pad=4)
            if c.size:
                z.writestr(f"logos/logo_{l['id']}.png", encode_png(c))
        for f in result["faces"]:
            c = crop(scene_bgr, f["box"], pad=int(0.15 * (f["box"][2] - f["box"][0])))
            if c.size:
                z.writestr(f"faces/{f['size'].lower()}/face_{f['id']}.png", encode_png(c))
        for o in result["objects"]:
            c = crop(scene_bgr, o["box"])
            if c.size:
                z.writestr(f"objects/{o['category'].replace(' & ', '_').replace(' ', '_')}/"
                           f"{o['class'].replace(' ', '_')}_{o['id']}.png", encode_png(c))
    return buf.getvalue()
