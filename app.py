"""
Hybrid AI Visual Search
-----------------------
Upload a large SOURCE image and (optionally) a small INPUT image such as a logo.
The app will
  * find every copy of the input image / logo inside the source image,
  * detect people's faces and categorise them,
  * detect and categorise other objects (YOLO).

Run:  streamlit run app.py
"""
import time
import traceback

import pandas as pd
import streamlit as st

from src.core.pipeline import run_pipeline
from src.utils.image_io import bgr_to_rgb, crop, encode_png, limit_size, load_pil, pil_to_bgr
from src.utils.report import build_zip, detections_frame, result_json

st.set_page_config(page_title="Hybrid AI Visual Search", page_icon="🔎", layout="wide")

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg,#312E81 0%,#4F46E5 45%,#0EA5E9 100%);}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .35rem 0 0 0; color: #E0E7FF; font-size: 1rem;}
.card {background:#111A2E; border:1px solid #223049; border-radius:14px; padding:1rem 1.1rem;}
.kpi {background:#111A2E; border:1px solid #223049; border-radius:14px; padding:.9rem 1rem; text-align:center;}
.kpi .n {font-size:2rem; font-weight:700; line-height:1.1;}
.kpi .l {font-size:.82rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.06em;}
.kpi.logo .n {color:#4ADE80;} .kpi.face .n {color:#FB923C;} .kpi.obj .n {color:#818CF8;} .kpi.t .n {color:#38BDF8;}
.tag {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem;
      background:#1E293B; border:1px solid #334155; margin:0 6px 6px 0;}
div[data-testid="stImage"] img {border-radius: 10px;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers
def show_image(img, caption=None, **kw):
    """st.image that works on both old and new Streamlit versions."""
    try:
        st.image(img, caption=caption, width="stretch", **kw)
    except TypeError:
        st.image(img, caption=caption, use_container_width=True, **kw)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def kpi(col, cls, number, label):
    col.markdown(f'<div class="kpi {cls}"><div class="n">{number}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def gallery(items, per_row=5):
    """items: list of (rgb_image, caption)."""
    for start in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (img, cap) in zip(cols, items[start:start + per_row]):
            with col:
                show_image(img, cap)


@st.cache_data(show_spinner=False, max_entries=8)
def _decode(data: bytes):
    return pil_to_bgr(load_pil(data))


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("**What should be detected?**")
    do_logo = st.toggle("Find the input image / logo", value=True)
    do_faces = st.toggle("Detect & categorise faces", value=True)
    do_objects = st.toggle("Detect & categorise objects (YOLO)", value=True)

    with st.expander("🔍 Logo / input-image search", expanded=False):
        use_features = st.checkbox("Feature matching (SIFT) – handles rotation", value=True)
        use_color = st.checkbox("Colour + shape matching – flat single-colour logos (e.g. McDonald's M)", value=True)
        color_tol = st.slider("Colour tolerance", 20, 80, 45,
                              help="Raise it if the logo in the photo is shaded, darker or washed out.")
        use_template = st.checkbox("Template matching – photo-like logos", value=True)
        min_inliers = st.slider("Min. matching points", 6, 40, 10,
                                help="Lower = finds weaker/blurrier matches but more false alarms.")
        template_thr = st.slider("Template threshold", 0.50, 0.95, 0.70, 0.01)
        max_instances = st.slider("Max. copies to find", 1, 30, 10)

    with st.expander("🙂 Faces", expanded=False):
        include_profile = st.checkbox("Include side-profile faces (more false positives)", value=False)
        face_conf = st.slider("DNN face confidence", 0.3, 0.95, 0.7, 0.05,
                              help="Only used when the YuNet model is available.")

    with st.expander("📦 Objects", expanded=False):
        yolo_model = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                                  help="n = fastest, m = most accurate. Downloaded on first use.")
        obj_conf = st.slider("Object confidence", 0.10, 0.90, 0.35, 0.05)

    max_side = st.select_slider("Max. working size (px, longest side)",
                                options=[2000, 3000, 4000, 6000, 8000], value=6000,
                                help="Very large images are scaled down to this size for speed.")
    st.divider()
    st.caption("Tip: crop the logo tightly, with little background, for the best results.")

# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>🔎 Hybrid AI Visual Search</h1>'
    "<p>Find a logo or image inside a large picture, detect people's faces and categorise everything it sees.</p></div>",
    unsafe_allow_html=True,
)

tab_run, tab_help = st.tabs(["🚀 Analyse", "📘 Help & Tips"])

with tab_run:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("##### 1 · Source image (big picture)")
        scene_file = st.file_uploader("Source image", type=["png", "jpg", "jpeg", "webp", "bmp"],
                                      key="scene", label_visibility="collapsed")
    with c2:
        st.markdown("##### 2 · Input image / logo (optional)")
        ref_file = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "webp", "bmp"],
                                    key="ref", label_visibility="collapsed")

    scene = reference = None
    try:
        if scene_file is not None:
            scene = _decode(scene_file.getvalue())
        if ref_file is not None:
            reference = _decode(ref_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read an image: {exc}")

    if scene is not None:
        p1, p2 = st.columns([3, 2], gap="large")
        with p1:
            show_image(bgr_to_rgb(scene), f"Source · {scene.shape[1]}×{scene.shape[0]} px")
        with p2:
            if reference is not None:
                show_image(bgr_to_rgb(reference), f"Input · {reference.shape[1]}×{reference.shape[0]} px")
            else:
                st.info("No input image uploaded – only faces and objects will be analysed.")

    if do_logo and reference is not None and not (use_features or use_template or use_color):
        st.warning("Enable at least one logo search method in the sidebar.")

    ready = scene is not None and (do_logo or do_faces or do_objects)
    run = st.button("▶  Run analysis", type="primary", disabled=not ready)

    if run and scene is not None:
        work_scene, s = limit_size(scene, max_side)
        status = st.status("Analysing image…", expanded=True)
        try:
            t0 = time.perf_counter()
            status.write("Preparing image" + (f" (scaled by {s:.2f})" if s < 1 else ""))
            result = run_pipeline(
                work_scene, reference,
                do_logo=do_logo and (use_features or use_template or use_color),
                do_faces=do_faces, do_objects=do_objects,
                logo_opts=dict(use_features=use_features, use_template=use_template,
                               use_color=use_color, color_tol=float(color_tol),
                               min_inliers=min_inliers, template_threshold=template_thr,
                               max_instances=max_instances),
                face_opts=dict(min_confidence=face_conf, include_profile=include_profile),
                obj_opts=dict(confidence=obj_conf, model_name=yolo_model),
            )
            status.update(label=f"Done in {time.perf_counter() - t0:.1f}s", state="complete", expanded=False)
            st.session_state["res_result"] = result
            st.session_state["res_scene"] = work_scene
            st.session_state["res_ran_with_ref"] = reference is not None
        except Exception as exc:
            status.update(label="Something went wrong", state="error")
            st.error(f"Could not complete the analysis: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

    # ----------------------------------------------------------------------- results
    result = st.session_state.get("res_result")
    work_scene = st.session_state.get("res_scene")
    if result is not None and work_scene is not None:
        st.divider()
        sm = result["summary"]
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "logo", sm["logos"], "Logo matches")
        kpi(k2, "face", sm["faces"], "Faces")
        kpi(k3, "obj", sm["objects"], "Objects")
        kpi(k4, "t", f"{result['timing_ms']['total'] / 1000:.1f}s", "Total time")
        st.write("")

        err = result["diagnostics"].get("objects", {}).get("error")
        if err:
            st.warning(err)
        if st.session_state.get("res_ran_with_ref") and not result["logos"] and "logo" in result["diagnostics"]:
            st.warning("The input image was not found. Try a tighter crop of the logo, lower "
                       "'Min. matching points', raise the colour tolerance, or lower the template threshold.")

        t_img, t_logo, t_face, t_obj, t_data = st.tabs(
            ["🖼️ Annotated", f"🏷️ Logo ({sm['logos']})", f"🙂 Faces ({sm['faces']})",
             f"📦 Objects ({sm['objects']})", "📊 Data & export"])

        with t_img:
            show_image(bgr_to_rgb(result["image"]))
            st.markdown('<span class="tag">🟩 logo</span><span class="tag">🟧 face</span>'
                        '<span class="tag">🟥 object</span>', unsafe_allow_html=True)
            st.download_button("⬇ Download annotated image", encode_png(result["image"]),
                               "annotated.png", "image/png")

        with t_logo:
            if not result["logos"]:
                st.info("No logo / input-image matches to show.")
            else:
                items = []
                for l in result["logos"]:
                    c = crop(work_scene, l["box"], pad=6)
                    if c.size:
                        items.append((bgr_to_rgb(c), f"#{l['id']} · {l['method']} · conf {l['confidence']:.2f}"))
                gallery(items, 4)
                show_table(pd.DataFrame([{"#": l["id"], "method": l["method"], "confidence": l["confidence"],
                                          "inliers": l.get("inliers", ""),
                                          "box (x1,y1,x2,y2)": [int(v) for v in l["box"]]}
                                         for l in result["logos"]]))

        with t_face:
            if not result["faces"]:
                st.info("No faces detected.")
            else:
                mc = st.columns(3)
                for col, size in zip(mc, ["Large", "Medium", "Small"]):
                    col.metric(f"{size} faces", sum(1 for f in result["faces"] if f["size"] == size))
                st.caption(f"Detector: {result['diagnostics']['faces']['method']}")
                for size in ["Large", "Medium", "Small"]:
                    group = [f for f in result["faces"] if f["size"] == size]
                    if not group:
                        continue
                    st.markdown(f"**{size} faces ({len(group)})**")
                    items = []
                    for f in group:
                        pad = int(0.15 * (f["box"][2] - f["box"][0]))
                        c = crop(work_scene, f["box"], pad=pad)
                        if c.size:
                            items.append((bgr_to_rgb(c), f"Face {f['id']} · {f['position']} · {f['pose']}"))
                    gallery(items, 6)

        with t_obj:
            if not result["objects"]:
                st.info("No objects detected (or YOLO is unavailable – see the warning above).")
            else:
                st.markdown("".join(f'<span class="tag">{cat}: <b>{n}</b></span>'
                                    for cat, n in sorted(sm["by_category"].items())),
                            unsafe_allow_html=True)
                cats = sorted({o["category"] for o in result["objects"]})
                for cat in cats:
                    group = [o for o in result["objects"] if o["category"] == cat]
                    with st.expander(f"{cat} ({len(group)})", expanded=cat == "People" or len(cats) <= 3):
                        items = []
                        for o in group[:30]:
                            c = crop(work_scene, o["box"])
                            if c.size:
                                items.append((bgr_to_rgb(c), f"{o['class']} · {o['confidence']:.2f}"))
                        gallery(items, 6)

        with t_data:
            df = detections_frame(result)
            if df.empty:
                st.info("Nothing detected.")
            else:
                show_table(df)
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇ CSV", df.to_csv(index=False), "detections.csv", "text/csv")
            d2.download_button("⬇ JSON report", result_json(result), "report.json", "application/json")
            d3.download_button("⬇ Everything (ZIP)", build_zip(result, work_scene),
                               "results.zip", "application/zip")
            with st.expander("Diagnostics"):
                st.json({"diagnostics": result["diagnostics"], "timing_ms": result["timing_ms"]})

with tab_help:
    st.markdown(
        """
### How it works
| Task | Method |
|---|---|
| **Find the input image / logo** | Three methods combined: SIFT features + RANSAC (scale, rotation, perspective), **colour + shape matching** (flat single-colour logos such as the McDonald's M, even on a different background colour) and multi-scale template matching. Several copies can be found. |
| **Faces** | OpenCV YuNet DNN if its small model can be downloaded, otherwise Haar cascades with confirmation filtering. Faces are categorised by **size** (Large / Medium / Small) and **position** (Left / Center / Right). |
| **Objects** | Ultralytics YOLOv8 (COCO, 80 classes) grouped into categories such as People, Vehicles, Animals, Electronics, Furniture… |

### Tips for the best results
* Crop the **input image / logo tightly** – little background around it.
* Use the **largest/cleanest** version of the logo you have.
* If the logo is **not found**, lower *Min. matching points* or the *Template threshold*.
* If you get **false matches**, raise them.
* Very large images are scaled down to the *Max. working size* for speed; coordinates in the
  exports refer to that working size.
* The first YOLO run downloads the model weights (~6 MB for `yolov8n`) and needs internet once.

### Limits
Face/logo detection is not perfect: tiny, blurred, heavily rotated or occluded items can be missed.
Face "categories" are descriptive (size/position); this app does **not** identify who a person is.
        """
    )