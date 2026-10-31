"""One-call pipeline: logo search + face detection + object categorization + annotation."""
import time
from collections import Counter

from .face_detection import detect_faces
from .logo_search import find_reference
from .object_detection import detect_objects
from ..utils.drawing import draw_box, draw_polygon


def run_pipeline(scene_bgr, ref_bgr=None, *, do_logo=True, do_faces=True, do_objects=True,
                 logo_opts=None, face_opts=None, obj_opts=None):
    t_all = time.perf_counter()
    out = scene_bgr.copy()
    result = {"logos": [], "faces": [], "objects": [], "diagnostics": {}, "timing_ms": {}}

    if do_logo and ref_bgr is not None:
        t = time.perf_counter()
        logos, diag = find_reference(scene_bgr, ref_bgr, **(logo_opts or {}))
        result["logos"], result["diagnostics"]["logo"] = logos, diag
        result["timing_ms"]["logo"] = round((time.perf_counter() - t) * 1000, 1)

    if do_objects:
        t = time.perf_counter()
        objs, info = detect_objects(scene_bgr, **(obj_opts or {}))
        result["objects"], result["diagnostics"]["objects"] = objs, info
        result["timing_ms"]["objects"] = round((time.perf_counter() - t) * 1000, 1)

    if do_faces:
        t = time.perf_counter()
        faces, info = detect_faces(scene_bgr, **(face_opts or {}))
        result["faces"], result["diagnostics"]["faces"] = faces, info
        result["timing_ms"]["faces"] = round((time.perf_counter() - t) * 1000, 1)

    # draw: objects first (background), then faces, then logos on top
    for o in result["objects"]:
        if o["class"] == "person" and do_faces:
            continue          # faces are drawn instead; person boxes stay in the table
        draw_box(out, o["box"], f"{o['class']} {o['confidence']:.2f}", "object")
    for f in result["faces"]:
        draw_box(out, f["box"], f"Face {f['id']}", "face")
    for i, l in enumerate(result["logos"], 1):
        l["id"] = i
        draw_polygon(out, l["polygon"], f"Logo {i} ({l['confidence']:.2f})", "logo")

    result["image"] = out
    result["summary"] = {
        "logos": len(result["logos"]),
        "faces": len(result["faces"]),
        "objects": len(result["objects"]),
        "by_category": dict(Counter(o["category"] for o in result["objects"])),
        "face_by_size": dict(Counter(f["size"] for f in result["faces"])),
    }
    result["timing_ms"]["total"] = round((time.perf_counter() - t_all) * 1000, 1)
    return result
