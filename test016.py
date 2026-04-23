#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test016.py - Cube Face-On Detector
-------------------------------------------------
Detect cube/box and determine if a face is front-facing

Default start equivalent to:

python test016.py --source 0 --conf 0.1 --min_area 1500 --color_thr 0.1 --show_all
    python test016.py --source 0 --roi 180,120,1100,600 --distance --known_width 10 --focal_length 800 --distance_unit cm

Criteria for "front-facing":
1. OBB aspect ratio close to 1:1 (square)
2. Sufficient area (not occluded)
3. Angle close to upright

Usage:
  python test016.py --source 0
  python test016.py --source video.mp4 --aspect_tol 0.3 --upright_tol 15

  python test016.py --source 0 --show_all 

  
  useful
  python test016.py --source 0 --conf 0.1 --min_area 1500 --color_thr 0.1  
"""
import sys
import os
import platform

# Set DISPLAY environment variable for Linux
if platform.system() == "Linux":
    os.environ["DISPLAY"] = ":0"
    print("Running on Linux - DISPLAY set to :0")

print("Loading modules...")
print("[1/6] argparse, pathlib, math", flush=True)
import argparse
from pathlib import Path
import math

print("[2/6] cv2", flush=True)
import cv2

print("[3/6] numpy", flush=True)
import numpy as np

print("[4/6] torch (this may take 30-60s on Surface Go)...", flush=True)
sys.stdout.flush()
try:
    import torch
    print(f"      torch {torch.__version__} loaded OK", flush=True)
except Exception as e:
    print(f"ERROR loading torch: {e}", flush=True)
    sys.exit(1)

print("[5/6] ultralytics", flush=True)
sys.stdout.flush()
try:
    from ultralytics import YOLOWorld
    print("      ultralytics loaded OK", flush=True)
except Exception as e:
    print(f"ERROR loading ultralytics: {e}", flush=True)
    sys.exit(1)

print("[6/6] All modules loaded successfully!\n", flush=True)

# ------------------------------- CLI ----------------------------------------

def build_args():
    ap = argparse.ArgumentParser("Cube face-on detector with red/blue check + distance")
    ap.add_argument("--source", type=str, default="0", help="Webcam only (must be 0)")
    ap.add_argument("--model", type=str, default="yolov8s-worldv2.pt", help="YOLO-World weights")
    ap.add_argument("--prompts", type=str, nargs="*", default=["box", "cube", "package", "container", "carton"],
                    help="Open-vocabulary prompts for box/cube objects")
    ap.add_argument("--imgsz", type=int, default=768, help="inference size")
    ap.add_argument("--conf", type=float, default=0.05, help="confidence threshold (default tuned for high sensitivity)")
    ap.add_argument("--max_det", type=int, default=12, help="maximum detections per frame")
    ap.add_argument("--min_area", type=float, default=800, help="minimum detection area (default smaller for sensitivity)")
    
    # Face-on criteria
    ap.add_argument("--aspect_tol", type=float, default=0.45, 
                    help="Tolerance for aspect ratio (1 +/- tol means square-ish, front-facing)")
    ap.add_argument("--upright_tol", type=float, default=28.0, 
                    help="Angle tolerance (deg) for upright orientation")
    ap.add_argument("--min_face_area", type=float, default=3200,
                    help="Minimum area for a valid front-facing detection")
    ap.add_argument("--extent_thr", type=float, default=0.62,
                    help="Min contour extent (area / minAreaRect area) to be considered front-facing")
    ap.add_argument("--poly_vertices_max", type=int, default=6,
                    help="Max polygon vertices for front-facing (approxPolyDP)")
    
    # Color filtering (red/blue)
    ap.add_argument("--color_mode", type=str, default="both",
                    choices=["red", "blue", "both", "off"],
                    help="Color filter mode (default both)")
    ap.add_argument("--color_thr", type=float, default=0.06, 
                    help="min fraction of red pixels to accept as red_box (default more sensitive)")
    ap.add_argument("--erode", type=int, default=5, 
                    help="pixels to shrink bbox before color check")
    ap.add_argument("--check_red", action="store_true", default=False,
                    help="Legacy: enable red-only mode (overrides color_mode if set)")
    
    # Distance estimation (from test012)
    ap.add_argument("--distance", action="store_true", default=True,
                    help="enable distance estimation overlays (default ON)")
    ap.add_argument("--distance_unit", type=str, default="cm", 
                    help="unit for distance display")
    ap.add_argument("--known_width", type=float, default=10.0, 
                    help="known real width of target object (in distance_unit)")
    ap.add_argument("--focal_length", type=float, default=800.0, 
                    help="camera focal length in pixels")
    
    # ROI (from test012)
    ap.add_argument("--roi", type=str, default="180,120,1100,600", 
                    help="ROI rectangle 'x1,y1,x2,y2' for detection-only zone (default set)")
    ap.add_argument("--roi_margin", type=int, default=0, 
                    help="Margin from all sides (ignored if --roi provided)")
    
    # Display options
    ap.add_argument("--save", action="store_true", help="save annotated frames")
    ap.add_argument("--outdir", type=str, default="cube_face_outputs", help="output directory")
    ap.add_argument("--headless", action="store_true", help="no display windows (implies --save)")
    ap.add_argument("--show_all", action="store_true", default=True,
                    help="Show all detections (not just front-facing) (default ON)")

    # Centering indicator
    ap.add_argument("--center_tol_frac", type=float, default=0.05,
                    help="Fraction of image width/height to consider centered")
    ap.add_argument("--center_pick", type=str, default="largest",
                    choices=["largest", "center", "conf"],
                    help="Which target drives centering indicator")
    ap.add_argument("--print_every", type=int, default=10,
                    help="Print guidance every N frames")
    
    return ap.parse_args()

# ----------------------------- Color utils (from test012) -------------------
HSV_RED1 = (np.array([0, 120, 70]), np.array([10, 255, 255]))
HSV_RED2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))
HSV_BLUE = (np.array([100, 150, 50]), np.array([140, 255, 255]))

def color_mask(hsv, color: str):
    if color == "red":
        return cv2.inRange(hsv, HSV_RED1[0], HSV_RED1[1]) | cv2.inRange(hsv, HSV_RED2[0], HSV_RED2[1])
    if color == "blue":
        return cv2.inRange(hsv, HSV_BLUE[0], HSV_BLUE[1])
    return None

def color_fraction(bgr_roi: np.ndarray, color: str) -> float:
    """Calculate fraction of color pixels in ROI"""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv, color)
    if mask is None:
        return 0.0
    total = bgr_roi.shape[0] * bgr_roi.shape[1]
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total)

def estimate_obbox_color(bgr_roi, color: str):
    """Get OBB + contour metrics from largest color contour in ROI"""
    if bgr_roi is None or bgr_roi.size == 0:
        return None
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv, color)
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(c)
    if contour_area < 8:
        return None
    rect = cv2.minAreaRect(c)
    (center, (w, h), angle) = rect
    angle_deg = float(angle)
    if w < h:
        angle_deg += 90.0
    if angle_deg < 0:
        angle_deg += 180.0
    box = cv2.boxPoints(rect).astype(int)
    rect_area = max(1.0, float(w * h))
    extent = float(contour_area) / rect_area
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) if peri > 0 else c
    vertices = int(len(approx)) if approx is not None else 0
    return box, angle_deg, extent, vertices

def shrink_xyxy(x1, y1, x2, y2, W, H, erode):
    """Shrink bounding box by erode pixels"""
    x1 = max(0, x1 + erode)
    y1 = max(0, y1 + erode)
    x2 = min(W - 1, x2 - erode)
    y2 = min(H - 1, y2 - erode)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def longest_side_length(pts):
    """Get longest side of 4-point polygon"""
    if pts is None or len(pts) != 4:
        return None
    d = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % 4]
        d.append(((x2 - x1)**2 + (y2 - y1)**2) ** 0.5)
    return max(d) if d else None

def dominant_edge_angle_deg(pts):
    """Return angle (deg) of the longest edge, normalized to [-90, 90]."""
    if pts is None or len(pts) != 4:
        return None
    best = None
    best_len = -1.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        dx, dy = (x2 - x1), (y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        if length > best_len:
            best_len = length
            best = (dx, dy)
    if best is None:
        return None
    angle = math.degrees(math.atan2(best[1], best[0]))
    # normalize to [-90, 90]
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle

# ----------------------------- Geometry utils -------------------------------

def get_obb_from_box(frame_bgr, x1, y1, x2, y2):
    """Extract oriented bounding box from detection region"""
    H, W = frame_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W-1, x2)
    y2 = min(H-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # Convert to grayscale and find edges
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Use largest contour
    c = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(c)
    if contour_area < 100:
        return None
    
    # Get minimum area rectangle
    rect = cv2.minAreaRect(c)
    (center, (w, h), angle) = rect
    
    # Normalize angle
    angle_deg = float(angle)
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    
    if angle_deg < 0:
        angle_deg += 180.0
    elif angle_deg >= 180:
        angle_deg -= 180.0
    
    # Get box points and offset to original image coordinates
    box = cv2.boxPoints(rect).astype(int)
    box[:, 0] += x1
    box[:, 1] += y1
    
    rect_area = max(1.0, float(w * h))
    extent = float(contour_area) / rect_area
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True) if peri > 0 else c
    vertices = int(len(approx)) if approx is not None else 0

    return {
        'box': box,
        'angle': angle_deg,
        'width': w,
        'height': h,
        'center': (center[0] + x1, center[1] + y1),
        'area': w * h,
        'extent': extent,
        'vertices': vertices
    }

def is_face_on(obb_info, args):
    """
    Determine if face is front-facing
    
    Criteria:
    1. Aspect ratio close to 1:1 (square face visible)
    2. Upright orientation (not heavily rotated)
    3. Sufficient area (not too far or occluded)
    """
    if obb_info is None:
        return False, 0.0
    
    w, h = obb_info['width'], obb_info['height']
    angle = obb_info['angle']
    area = obb_info['area']
    
    # Check minimum area
    if area < args.min_face_area:
        return False, 0.0
    
    # Check aspect ratio (should be close to 1:1 for square face)
    aspect = w / h if h > 0 else 0
    aspect_dev = abs(aspect - 1.0)
    aspect_ok = aspect_dev <= args.aspect_tol
    
    # Check upright (angle close to 0 or 90 or 180)
    upright_tol = args.upright_tol
    angle_mod = angle % 90
    upright_ok = angle_mod <= upright_tol or angle_mod >= (90 - upright_tol)
    
    extent = obb_info.get('extent', None)
    vertices = obb_info.get('vertices', None)
    extent_ok = True
    vertices_ok = True
    if extent is not None:
        extent_ok = extent >= args.extent_thr
    if vertices is not None and args.poly_vertices_max > 0:
        vertices_ok = vertices <= args.poly_vertices_max

    # Confidence score (0-1)
    aspect_score = max(0, 1.0 - aspect_dev / args.aspect_tol)
    angle_score = 1.0 - min(angle_mod, 90 - angle_mod) / 45.0
    area_score = min(1.0, area / (args.min_face_area * 2))
    extent_score = min(1.0, max(0.0, extent)) if extent is not None else 0.5
    vertices_score = 1.0 if (vertices is None or vertices <= args.poly_vertices_max) else 0.0

    confidence = (aspect_score * 0.4 + angle_score * 0.2 + area_score * 0.2 + extent_score * 0.1 + vertices_score * 0.1)
    
    is_front = aspect_ok and upright_ok and extent_ok and vertices_ok
    
    return is_front, confidence

# ---------------------------- I/O helpers -----------------------------------

def iter_sources(src: str):
    if src != "0":
        raise ValueError("This program supports webcam only. Use --source 0.")
    yield "webcam", None

# ------------------------------ Main ----------------------------------------

def main():
    args = build_args()
    if args.headless:
        args.save = True
    if args.check_red and args.color_mode == "both":
        args.color_mode = "red"

    def apply_color_mode(mode: str):
        if mode == "red":
            args.red_on, args.blue_on = True, False
        elif mode == "blue":
            args.red_on, args.blue_on = False, True
        elif mode == "both":
            args.red_on, args.blue_on = True, True
        else:
            args.red_on, args.blue_on = False, False

    apply_color_mode(args.color_mode)
    args._last_toggle_state = (args.red_on, args.blue_on)

    print(f"Loading YOLO-World model: {args.model} ...")
    sys.stdout.flush()
    model = YOLOWorld(args.model)
    model.set_classes(args.prompts)
    print("Model ready!\n")

    outdir = Path(args.outdir)
    if args.save:
        (outdir / "frames").mkdir(parents=True, exist_ok=True)
        (outdir / "front_facing").mkdir(parents=True, exist_ok=True)

    window_name = "Cube Face-On Detector (Press q to Quit)"
    for kind, src in iter_sources(args.source):
        if kind == "webcam":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open webcam")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not args.headless:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1280, 720)
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                vis, status_text = process_frame(model, frame, args, idx)
                if args.save:
                    cv2.imwrite(str(outdir / "frames" / f"frame_{idx:06d}.jpg"), vis)
                if not args.headless:
                    cv2.imshow(window_name, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    if key in (ord('r'), ord('R')):
                        args.red_on = not args.red_on
                    if key in (ord('b'), ord('B')):
                        args.blue_on = not args.blue_on
                    if (args.red_on, args.blue_on) != args._last_toggle_state:
                        args._last_toggle_state = (args.red_on, args.blue_on)
                        print(f"[MODE] red={'ON' if args.red_on else 'OFF'} blue={'ON' if args.blue_on else 'OFF'}")
                if status_text and (idx % max(1, int(args.print_every)) == 0):
                    print(status_text)
                idx += 1
            cap.release()
            if not args.headless:
                cv2.destroyAllWindows()
        else:
            if is_video(src):
                cap = cv2.VideoCapture(0 if src.isdigit() else src)
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    vis, status_text = process_frame(model, frame, args, idx)
                    if args.save:
                        cv2.imwrite(str(outdir / "frames" / f"frame_{idx:06d}.jpg"), vis)
                    if not args.headless:
                        cv2.imshow("Cube Face-On Detector (press q to quit)", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord('q'), ord('Q')):
                            break
                        if key in (ord('r'), ord('R')):
                            args.red_on = not args.red_on
                        if key in (ord('b'), ord('B')):
                            args.blue_on = not args.blue_on
                        if (args.red_on, args.blue_on) != args._last_toggle_state:
                            args._last_toggle_state = (args.red_on, args.blue_on)
                            print(f"[MODE] red={'ON' if args.red_on else 'OFF'} blue={'ON' if args.blue_on else 'OFF'}")
                    if status_text and (idx % max(1, int(args.print_every)) == 0):
                        print(status_text)
                    idx += 1
                cap.release()
                if not args.headless:
                    cv2.destroyAllWindows()
            else:
                img = cv2.imread(src)
                if img is None:
                    print(f"Read failed: {src}")
                    continue
                vis, status_text = process_frame(model, img, args, 0)
                if args.save:
                    cv2.imwrite(str(outdir / "frames" / f"{Path(src).stem}_pred.jpg"), vis)
                if status_text:
                    print(status_text)
                if not args.headless:
                    cv2.imshow("Cube Face-On Detector (press any key to continue)", vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

# ------------------------- Frame processing ---------------------------------

def process_frame(model: YOLOWorld, frame_bgr: np.ndarray, args, frame_idx: int):
    H, W = frame_bgr.shape[:2]
    vis = frame_bgr.copy()
    guidance_text = ""
    
    # --- ROI processing (from test012) ---
    roi_rect = None
    if getattr(args, 'roi', None) and str(args.roi).strip():
        try:
            parts = [int(v) for v in str(args.roi).split(',')]
            if len(parts) == 4:
                rx1, ry1, rx2, ry2 = parts
                rx1 = max(0, min(W - 1, rx1))
                ry1 = max(0, min(H - 1, ry1))
                rx2 = max(0, min(W - 1, rx2))
                ry2 = max(0, min(H - 1, ry2))
                if rx2 > rx1 and ry2 > ry1:
                    roi_rect = (rx1, ry1, rx2, ry2)
        except Exception:
            roi_rect = None
    if roi_rect is None and getattr(args, 'roi_margin', 0) and int(args.roi_margin) > 0:
        m = int(args.roi_margin)
        rx1, ry1 = m, m
        rx2, ry2 = W - m, H - m
        if rx2 > rx1 and ry2 > ry1:
            roi_rect = (rx1, ry1, rx2, ry2)
    
    # ROI border removed (webcam-only, no detection area box)
    
    # Run YOLO-World detection
    res = model.predict(frame_bgr, imgsz=args.imgsz, conf=args.conf, 
                       max_det=args.max_det, verbose=False)[0]
    
    if getattr(res, 'boxes', None) is None or res.boxes is None:
        # Draw status text
        cv2.putText(vis, "No detections", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return vis
    
    names = res.names if isinstance(res.names, dict) else {}
    
    front_facing_count = 0
    red_count = 0
    blue_count = 0
    detections = []
    
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].int().tolist()
        area = (x2 - x1) * (y2 - y1)
        
        if area < args.min_area:
            continue
        
        # --- ROI center check (from test012) ---
        if roi_rect is not None:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx1, ry1, rx2, ry2 = roi_rect
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                cv2.rectangle(vis, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue
        
        cls = int(b.cls) if hasattr(b, 'cls') and b.cls is not None else -1
        name = names.get(cls, str(cls))
        conf = float(b.conf) if hasattr(b, 'conf') and b.conf is not None else 0.0
        
        # --- Red/Blue color check ---
        if not getattr(args, "red_on", True) and not getattr(args, "blue_on", True):
            continue
        is_red = False
        is_blue = False
        frac_red = 0.0
        frac_blue = 0.0
        shr = shrink_xyxy(x1, y1, x2, y2, W, H, args.erode)
        if shr is not None:
            sx1, sy1, sx2, sy2 = shr
            roi = frame_bgr[sy1:sy2, sx1:sx2]
            if getattr(args, "red_on", False):
                frac_red = color_fraction(roi, "red")
                is_red = frac_red >= args.color_thr
            if getattr(args, "blue_on", False):
                frac_blue = color_fraction(roi, "blue")
                is_blue = frac_blue >= args.color_thr
        if is_red:
            red_count += 1
        if is_blue:
            blue_count += 1
        if not (is_red or is_blue):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (160, 160, 160), 1)
            continue

        if is_red and is_blue:
            color_label = "RED" if frac_red >= frac_blue else "BLUE"
        elif is_red:
            color_label = "RED"
        else:
            color_label = "BLUE"
        
        # Get oriented bounding box (use color-based OBB if possible, else edge-based)
        obb_info = None
        if color_label and shr:
            sx1, sy1, sx2, sy2 = shr
            roi = frame_bgr[sy1:sy2, sx1:sx2]
            obb_result = estimate_obbox_color(roi, color_label.lower())
            if obb_result:
                obb, angle, extent, vertices = obb_result
                obb_img = obb + np.array([sx1, sy1])
                w_h = longest_side_length(obb)
                h_h = np.linalg.norm(obb[1] - obb[0])
                if w_h and h_h:
                    obb_info = {
                        'box': obb_img,
                        'angle': angle,
                        'width': max(w_h, h_h),
                        'height': min(w_h, h_h),
                        'center': ((sx1 + sx2)/2, (sy1 + sy2)/2),
                        'area': w_h * h_h if w_h and h_h else area,
                        'extent': extent,
                        'vertices': vertices
                    }
        
        if obb_info is None:
            # Fallback to edge-based OBB
            obb_info = get_obb_from_box(frame_bgr, x1, y1, x2, y2)
        
        # Check if face-on
        is_front, face_conf = is_face_on(obb_info, args)
        
        # --- Distance estimation (from test012) ---
        Z = None
        dist_center = None
        if getattr(args, 'distance', False):
            f = float(getattr(args, 'focal_length', 0.0) or 0.0)
            W_real = float(getattr(args, 'known_width', 0.0) or 0.0)
            if f > 0.0 and W_real > 0.0:
                if obb_info is not None:
                    W_pix = longest_side_length(obb_info['box'])
                else:
                    W_pix = float(x2 - x1)
                if W_pix and W_pix > 1.0:
                    Z = (W_real * f) / float(W_pix)
                    cx_pix = (x1 + x2) / 2.0
                    cy_pix = (y1 + y2) / 2.0
                    cx_cam, cy_cam = W / 2.0, H / 2.0
                    X = (cx_pix - cx_cam) * Z / f
                    Y = (cy_pix - cy_cam) * Z / f
                    dist_center = math.sqrt(X*X + Y*Y + Z*Z)
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'obb': obb_info,
            'is_front': is_front,
            'is_red': is_red,
            'is_blue': is_blue,
            'color': color_label,
            'confidence': face_conf,
            'name': name,
            'yolo_conf': conf,
            'Z': Z,
            'dist': dist_center,
            'area': obb_info['area'] if obb_info and 'area' in obb_info else area,
            'center': obb_info['center'] if obb_info and 'center' in obb_info else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        })
        
        if is_front:
            front_facing_count += 1
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        obb_info = det['obb']
        is_front = det['is_front']
        is_red = det['is_red']
        is_blue = det['is_blue']
        face_conf = det['confidence']
        Z = det['Z']
        dist_center = det['dist']
        
        # Choose color: GREEN for front-facing, RED for angled
        color = (0, 255, 0) if is_front else (0, 0, 255)
        thickness = 3 if is_front else 2
        
        # Draw all when --check_red is on; otherwise only draw front-facing unless --show_all
        if getattr(args, 'check_red', False) or is_front or args.show_all:
            if obb_info is not None:
                # Draw OBB
                cv2.polylines(vis, [obb_info['box']], True, color, thickness)
                
                # Draw center point
                cx, cy = int(obb_info['center'][0]), int(obb_info['center'][1])
                cv2.circle(vis, (cx, cy), 5, color, -1)
                
                # Prepare label
                status = "FRONT-FACING" if is_front else "ANGLED"
                if is_red and is_blue:
                    status += " [RED+BLUE]"
                elif is_red:
                    status += " [RED]"
                elif is_blue:
                    status += " [BLUE]"
                label = f"{status} | {det['name']}"
                
                # Additional info
                aspect = obb_info['width'] / obb_info['height'] if obb_info['height'] > 0 else 0
                extent = obb_info.get('extent', None)
                vertices = obb_info.get('vertices', None)
                info_text = f"Aspect: {aspect:.2f} | Angle: {obb_info['angle']:.1f} deg | Conf: {face_conf:.2f}"
                if extent is not None:
                    info_text += f" | Ext: {extent:.2f}"
                if vertices is not None:
                    info_text += f" | V: {vertices:d}"
                
                # Distance info
                if Z is not None and Z > 0:
                    unit = getattr(args, 'distance_unit', 'cm')
                    info_text += f" | Z={Z:.1f}{unit}"
                    if dist_center is not None:
                        info_text += f" D={dist_center:.1f}{unit}"
                
                # Draw labels
                rx, ry, rw, rh = cv2.boundingRect(obb_info['box'])
                y_label = max(30, ry - 10)
                y_info = max(50, ry - 30)
                
                cv2.putText(vis, label, (rx, y_label), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Use sharper magenta color for info text (avoid white)
                cv2.putText(vis, info_text, (rx, y_info), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Save front-facing crops
                if args.save and is_front:
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_path = Path(args.outdir) / "front_facing" / f"front_{frame_idx:06d}_{x1}_{y1}.jpg"
                        cv2.imwrite(str(crop_path), crop)
            else:
                # Fallback: draw regular bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                label = f"{'FRONT' if is_front else 'Angled'} | {det['name']}"
                if is_red and is_blue:
                    label += " [RED+BLUE]"
                elif is_red:
                    label += " [RED]"
                elif is_blue:
                    label += " [BLUE]"
                cv2.putText(vis, label, (x1, max(20, y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Centering indicator: four lines to screen edges + center box
    center_x, center_y = W / 2.0, H / 2.0
    cx, cy = int(center_x), int(center_y)
    cv2.line(vis, (0, cy), (W, cy), (255, 255, 0), 2)
    cv2.line(vis, (cx, 0), (cx, H), (255, 255, 0), 2)
    box_w = int(float(args.center_tol_frac) * W * 2)
    box_h = int(float(args.center_tol_frac) * H * 2)
    box_w = max(20, box_w)
    box_h = max(20, box_h)
    cv2.rectangle(vis, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2),
                  (255, 255, 0), 2)

    front_targets = [d for d in detections if d['is_front']]
    any_targets = detections
    target = None
    pool = front_targets if front_targets else any_targets
    if pool:
        if args.center_pick == "largest":
            target = max(pool, key=lambda d: d['area'])
        elif args.center_pick == "center":
            target = min(pool, key=lambda d: (d['center'][0] - center_x) ** 2 + (d['center'][1] - center_y) ** 2)
        else:
            target = max(pool, key=lambda d: d['yolo_conf'])

    if target is None:
        guidance_text = "NO TARGET"
        cv2.putText(vis, guidance_text, (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        tx, ty = target['center']
        dx = tx - center_x
        dy = ty - center_y
        tol_x = float(args.center_tol_frac) * W
        tol_y = float(args.center_tol_frac) * H

        if abs(dx) <= tol_x and abs(dy) <= tol_y:
            guidance_text = "CENTERED"
        else:
            dirs = []
            if abs(dx) > tol_x:
                dirs.append("RIGHT" if dx > 0 else "LEFT")
            if abs(dy) > tol_y:
                dirs.append("DOWN" if dy > 0 else "UP")
            guidance_text = "MOVE " + " ".join(dirs)

        cv2.circle(vis, (int(tx), int(ty)), 6, (255, 255, 0), -1)
        cv2.putText(vis, f"{guidance_text} dx={dx:.0f} dy={dy:.0f}", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Angle correction indicator when angled
        if not target['is_front'] and target['obb'] is not None:
            edge_angle = dominant_edge_angle_deg(target['obb']['box'])
            if edge_angle is not None:
                rotate_dir = "LEFT" if edge_angle > 0 else "RIGHT"
                arrow_len = 60
                if rotate_dir == "LEFT":
                    p1 = (int(tx + arrow_len / 2), int(ty))
                    p2 = (int(tx - arrow_len / 2), int(ty))
                else:
                    p1 = (int(tx - arrow_len / 2), int(ty))
                    p2 = (int(tx + arrow_len / 2), int(ty))
                cv2.arrowedLine(vis, p1, p2, (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(vis, f"ROTATE {rotate_dir}", (int(tx) - 80, int(ty) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw status bar
    status_bar_text = f"Front-Facing: {front_facing_count}"
    if getattr(args, "red_on", False):
        status_bar_text += f" | Red: {red_count}"
    if getattr(args, "blue_on", False):
        status_bar_text += f" | Blue: {blue_count}"
    status_bar_text += f" / {len(detections)} total"
    
    mode_text = f"Mode: Red={'ON' if getattr(args, 'red_on', False) else 'OFF'} Blue={'ON' if getattr(args, 'blue_on', False) else 'OFF'}"
    cv2.rectangle(vis, (10, 10), (W - 10, 100), (0, 0, 0), -1)
    cv2.rectangle(vis, (10, 10), (W - 10, 100), (0, 255, 255), 2)
    cv2.putText(vis, status_bar_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, mode_text, (20, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return vis, guidance_text

if __name__ == "__main__":
    main()
