"""
Find a small reference image (logo / object) inside a large source image.

Two complementary strategies are combined:
  1. Local-feature matching (SIFT, falls back to ORB) + RANSAC homography.
     Handles scale change, rotation, and moderate perspective. Finds several copies.
  2. Multi-scale normalized template matching (+ NMS).
     Handles flat logos with too few keypoints, but is not rotation invariant.
Results from both are merged and de-duplicated.
"""
import cv2
import numpy as np

# ----------------------------------------------------------------------------- helpers

def _box_from_polygon(poly):
    xs, ys = poly[:, 0], poly[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def box_iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _resize_max(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img, s


def _prep_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _is_valid_quad(poly, scene_w, scene_h):
    """Reject folded / degenerate / absurd homographies."""
    if not np.all(np.isfinite(poly)):
        return False
    if not cv2.isContourConvex(np.float32(poly).reshape(-1, 1, 2)):
        return False
    area = abs(cv2.contourArea(np.float32(poly)))
    if area < 0.00005 * scene_w * scene_h or area > 0.95 * scene_w * scene_h:
        return False
    # edge lengths must be comparable (no needle-like shapes)
    edges = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
    if min(edges) < 4 or max(edges) / max(min(edges), 1e-6) > 12:
        return False
    # mostly inside the scene
    margin_x, margin_y = 0.15 * scene_w, 0.15 * scene_h
    if (poly[:, 0].min() < -margin_x or poly[:, 0].max() > scene_w + margin_x or
            poly[:, 1].min() < -margin_y or poly[:, 1].max() > scene_h + margin_y):
        return False
    return True


def _nms(dets, iou_thr):
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        if all(box_iou_xyxy(d["box"], k["box"]) < iou_thr for k in kept):
            kept.append(d)
    return kept

# ----------------------------------------------------------------------------- feature matching

def _make_detector(kind, big):
    if kind == "SIFT" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02 if big else 0.03), cv2.NORM_L2, "SIFT"
    return cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=12, fastThreshold=10), cv2.NORM_HAMMING, "ORB"


def _match_instances(kp_r, des_r, ref_shape, kp_s, des_s, scene_shape, norm, ratio,
                     min_inliers, ransac_thr, max_instances, used, s_scale):
    matcher = cv2.BFMatcher(norm)
    ph, pw = ref_shape[:2]
    sh, sw = scene_shape[:2]
    corners = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]]).reshape(-1, 1, 2)
    alive = np.ones(len(kp_s), dtype=bool)
    detections, best_good = [], 0
    for _ in range(max_instances):
        idx = np.where(alive)[0]
        if len(idx) < max(min_inliers, 4):
            break
        pairs = matcher.knnMatch(des_r, des_s[idx], k=2)
        good = [m for p in pairs if len(p) == 2 for m, n in [p] if m.distance < ratio * n.distance]
        best_good = max(best_good, len(good))
        if len(good) < min_inliers:
            break
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[idx[m.trainIdx]].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thr, maxIters=3000, confidence=0.995)
        if H is None or mask is None:
            break
        mask = mask.ravel().astype(bool)
        inl = int(mask.sum())
        if inl < min_inliers:
            break
        poly = cv2.perspectiveTransform(corners, H).reshape(4, 2)
        for m, ok in zip(good, mask):          # consume inliers so the next loop finds another copy
            if ok:
                alive[idx[m.trainIdx]] = False
        quad = np.int32(poly).reshape(-1, 1, 2)
        for i in np.where(alive)[0]:
            if cv2.pointPolygonTest(quad, tuple(map(float, kp_s[i].pt)), False) >= 0:
                alive[i] = False
        if not _is_valid_quad(poly, sw, sh):
            continue
        conf = float(min(1.0, 0.5 * min(inl / 40.0, 1.0) + 0.5 * inl / len(good)))
        poly_full = poly / s_scale
        detections.append({"method": f"Feature ({used})", "box": _box_from_polygon(poly_full),
                           "polygon": poly_full.tolist(), "confidence": round(conf, 4),
                           "inliers": inl, "good_matches": len(good)})
    return detections, best_good


def feature_search(scene_bgr, ref_bgr, detector="SIFT", ratio=0.78, min_inliers=10,
                   ransac_thr=4.0, max_instances=10, scene_max_side=3200):
    """Local-feature search. The reference is tried at several sizes, because a big logo
    must be shrunk to the (often tiny) size it has inside a large scene."""
    scene_small, s_scale = _resize_max(scene_bgr, scene_max_side)
    scene_gray = _prep_gray(scene_small)
    det, norm, used = _make_detector(detector, big=max(scene_gray.shape) > 1500)
    kp_s, des_s = det.detectAndCompute(scene_gray, None)

    rh, rw = ref_bgr.shape[:2]
    longest = max(rh, rw)
    sizes = [200] if longest < 200 else sorted({min(longest, t) for t in (400, 220, 120)}, reverse=True)
    info = {"detector": used, "scene_keypoints": len(kp_s or []), "ref_sizes_tried": sizes,
            "status": ""}
    if des_s is None or len(kp_s) < 4:
        info["status"] = "Source image has too few distinctive features."
        return [], info

    all_det, best_good, ref_kp = [], 0, 0
    for size in sizes:
        k = size / float(longest)
        ref = cv2.resize(ref_bgr, (max(8, int(round(rw * k))), max(8, int(round(rh * k)))),
                         interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        ref_gray = _prep_gray(ref)
        kp_r, des_r = det.detectAndCompute(ref_gray, None)
        ref_kp = max(ref_kp, len(kp_r or []))
        if des_r is None or len(kp_r) < 4:
            continue
        d, bg = _match_instances(kp_r, des_r, ref_gray.shape, kp_s, des_s, scene_gray.shape, norm,
                                 ratio, min_inliers, ransac_thr, max_instances, used, s_scale)
        all_det += d
        best_good = max(best_good, bg)
    detections = _nms(all_det, 0.3)[:max_instances]
    info["ref_keypoints"] = ref_kp
    info["best_good_matches"] = best_good
    if detections:
        info["status"] = f"{len(detections)} instance(s) located"
    elif ref_kp < 20:
        info["status"] = "Reference has few distinctive features (flat logo) - colour/template matching is used instead."
    else:
        info["status"] = "No geometrically consistent match found"
    return detections, info

# ----------------------------------------------------------------------------- template matching

def _size_scales(ref_shape, scene_shape, steps):
    """Scales so the reference's longest side runs from ~20 px up to 90% of the scene.
    (A 1280 px logo may appear as only ~80 px inside the scene.)"""
    rh, rw = ref_shape[:2]
    sh, sw = scene_shape[:2]
    lo = 20.0 / max(rh, rw)
    hi = 0.9 * min(sh / float(rh), sw / float(rw))
    if hi <= lo:
        return []
    return list(np.geomspace(lo, hi, int(steps)))


def _suppress_peaks(resp, threshold, k):
    peaks = (resp >= threshold) & (resp == cv2.dilate(resp, np.ones((k, k), np.uint8)))
    return np.where(peaks)


def template_search(scene_bgr, ref_bgr, threshold=0.70, steps=45, max_instances=10,
                    nms_iou=0.30, scene_max_side=2200):
    scene_small, s_scale = _resize_max(scene_bgr, scene_max_side)
    scene_gray = _prep_gray(scene_small)
    ref_gray = _prep_gray(ref_bgr)
    sh, sw = scene_gray.shape[:2]
    ref_w = cv2.resize(ref_gray, None, fx=s_scale, fy=s_scale, interpolation=cv2.INTER_AREA) \
        if s_scale < 1.0 else ref_gray

    cands, tested = [], 0
    for sc in _size_scales(ref_w.shape, scene_gray.shape, steps):
        w, h = int(round(ref_w.shape[1] * sc)), int(round(ref_w.shape[0] * sc))
        if w < 12 or h < 12 or w >= sw or h >= sh:
            continue
        tpl = cv2.resize(ref_w, (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        if tpl.std() < 3:
            continue
        tested += 1
        resp = np.nan_to_num(cv2.matchTemplate(scene_gray, tpl, cv2.TM_CCOEFF_NORMED),
                             nan=0.0, posinf=0.0, neginf=0.0)
        ys, xs = _suppress_peaks(resp, threshold, max(3, (min(w, h) // 2) | 1))
        for y, x in list(zip(ys, xs))[:200]:
            box = [x / s_scale, y / s_scale, (x + w) / s_scale, (y + h) / s_scale]
            cands.append({"method": "Template", "box": box,
                          "polygon": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                          "confidence": round(float(resp[y, x]), 4)})
    kept = _nms(cands, nms_iou)[:max_instances]
    info = {"scales_tested": tested, "threshold": threshold,
            "best_score": max((d["confidence"] for d in kept), default=0.0),
            "status": f"{len(kept)} instance(s) located" if kept else "No match above threshold"}
    return kept, info

# ----------------------------------------------------------------------------- colour + shape matching

def _reference_shape(ref_bgr):
    """Split a (flat) logo from its background. Returns (binary shape crop, Lab colour) or None."""
    small, _ = _resize_max(ref_bgr, 400)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(lab - bg, axis=2) > 35
    frac = float(fg.mean())
    if frac < 0.02 or frac > 0.85:
        return None
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    shape = fg[y0:y1, x0:x1].astype(np.float32)
    color = np.median(lab[fg], axis=0)
    # only for genuinely single-colour logos; multi-colour logos would match just one part
    purity = float((np.linalg.norm(lab[fg] - color, axis=1) < 40).mean())
    if purity < 0.75:
        return None
    if shape.shape[0] < 6 or shape.shape[1] < 6:
        return None
    return shape, color


def color_shape_search(scene_bgr, ref_bgr, color_tol=45.0, threshold=0.55, steps=50,
                       max_instances=10, nms_iou=0.30, scene_max_side=2200):
    """Find a flat-colour logo (e.g. yellow arches) regardless of what background it sits on.
    The reference's colour is turned into a soft mask of the scene and the logo's silhouette is
    matched against that mask at many sizes. Works when grayscale/feature matching cannot,
    because the logo may appear on red, brown or white backgrounds."""
    ref = _reference_shape(ref_bgr)
    if ref is None:
        return [], {"status": "Reference is not a single-colour logo on a plain background - skipped."}
    shape, color = ref
    scene_small, s_scale = _resize_max(scene_bgr, scene_max_side)
    lab = cv2.cvtColor(scene_small, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - color, axis=2)
    soft = np.clip(1.0 - dist / (color_tol * 1.6), 0.0, 1.0).astype(np.float32)
    soft = cv2.GaussianBlur(soft, (3, 3), 0)
    sh, sw = soft.shape
    cov = float((dist < color_tol).mean())
    if cov < 1e-5:
        return [], {"status": "The logo colour does not appear in the source image.", "colour_coverage": cov}

    cands, tested = [], 0
    ref_scaled_shape = shape.shape                       # compute scales relative to the shape crop
    work_shape = shape
    for sc in _size_scales(ref_scaled_shape, soft.shape, steps):
        w, h = int(round(shape.shape[1] * sc)), int(round(shape.shape[0] * sc))
        if w < 10 or h < 10 or w >= sw or h >= sh:
            continue
        tpl = cv2.resize(work_shape, (w, h), interpolation=cv2.INTER_AREA)
        tpl = cv2.GaussianBlur(tpl, (3, 3), 0)
        if tpl.std() < 0.05:
            continue
        tested += 1
        resp = np.nan_to_num(cv2.matchTemplate(soft, tpl, cv2.TM_CCOEFF_NORMED),
                             nan=0.0, posinf=0.0, neginf=0.0)
        # the logo must really be made of the logo colour: >= 45% of the silhouette area covered
        fill = cv2.matchTemplate(soft, (tpl > 0.5).astype(np.float32), cv2.TM_CCORR) / max(1.0, float((tpl > 0.5).sum()))
        ys, xs = _suppress_peaks(np.where(fill >= 0.45, resp, 0.0), threshold, max(3, (min(w, h) // 2) | 1))
        for y, x in list(zip(ys, xs))[:200]:
            box = [x / s_scale, y / s_scale, (x + w) / s_scale, (y + h) / s_scale]
            cands.append({"method": "Colour-shape", "box": box,
                          "polygon": [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                          "confidence": round(float(resp[y, x]), 4)})
    kept = _nms(cands, nms_iou)[:max_instances]
    info = {"scales_tested": tested, "threshold": threshold, "colour_coverage": round(cov, 5),
            "best_score": max((d["confidence"] for d in kept), default=0.0),
            "status": f"{len(kept)} instance(s) located" if kept else "No colour-shape match above threshold"}
    return kept, info

# ----------------------------------------------------------------------------- combined

def find_reference(scene_bgr, ref_bgr, use_features=True, use_template=True, use_color=True,
                   detector="SIFT", feature_ratio=0.78, min_inliers=10, template_threshold=0.70,
                   color_tol=45.0, color_threshold=0.55, max_instances=10):
    """Return (detections, diagnostics). Each detection has box [x1,y1,x2,y2] in scene pixels."""
    if scene_bgr is None or ref_bgr is None or scene_bgr.size == 0 or ref_bgr.size == 0:
        raise ValueError("Both the source image and the reference image are required.")
    ref_bgr, _ = _resize_max(ref_bgr, 900)       # huge references only cost memory/time
    diag, found = {}, []

    def inside(a, b):
        """Fraction of box a that lies inside box b."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))

    def add_new(dets):
        boxes = [x["box"] for x in found]
        return [x for x in dets if all(box_iou_xyxy(x["box"], b) < 0.3 and inside(x["box"], b) < 0.5
                                       for b in boxes)]

    if use_features:
        d, info = feature_search(scene_bgr, ref_bgr, detector=detector, ratio=feature_ratio,
                                 min_inliers=min_inliers, max_instances=max_instances)
        found += d
        diag["feature_matching"] = info
    if use_color:
        d, info = color_shape_search(scene_bgr, ref_bgr, color_tol=color_tol,
                                     threshold=color_threshold, max_instances=max_instances)
        found += add_new(d)
        diag["colour_shape_matching"] = info
    if use_template:
        d, info = template_search(scene_bgr, ref_bgr, threshold=template_threshold,
                                  max_instances=max_instances)
        found += add_new(d)
        diag["template_matching"] = info
    rank = {"Feature": 2.0, "Colour-shape": 1.0, "Template": 0.0}
    found.sort(key=lambda x: rank.get(x["method"].split(" ")[0], 0) + x["confidence"], reverse=True)
    return found[:max_instances], diag