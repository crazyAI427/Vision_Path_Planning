#!/usr/bin/env python
"""Autonomous path planning overlay.

Visual output (single frame):
  Green   overlay   = drivable road region
  Cyan    band      = passable corridor (bottom portion only, ≥ car width)
  White   dots      = short centerline guideline (near-field only)
  Orange  segments  = obstacles that are in / adjacent to the road corridor
  HUD               = CLEAR / SLOW / DETOUR LEFT·RIGHT / STOP

No bearing line. No blue/red dots.
Obstacles shown as segmented regions, not bounding boxes.

Usage
─────
python drive_direction.py --input input/
python drive_direction.py --input test.mp4
python drive_direction.py --input test.mp4 --car-width 0.20
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from inference import infer_seg, load_model

# ── Cityscapes trainId sets ───────────────────────────────────────────────────
ROAD_CLASSES     = {0, 9}
OBSTACLE_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}  # person…bicycle

# ── parameters ────────────────────────────────────────────────────────────────
CAR_WIDTH_FRAC  = 0.18   # vehicle width as fraction of image width
GUIDE_FRAC      = 0.20   # show corridor/centerline only in bottom GUIDE_FRAC of frame
LOOKAHEAD_FRAC  = 0.15   # fraction into passable rows (0=far, 1=near) for steering
CLAMP_MARGIN    = 10     # px inset from corridor edge
STOP_ZONE_FRAC  = 0.28   # bottom fraction → obstacle here = STOP
WARN_ZONE_FRAC  = 0.52   # bottom fraction → obstacle here = SLOW
MIN_OBS_AREA    = 150    # px² noise threshold

# ── colours (BGR) ─────────────────────────────────────────────────────────────
C_ROAD      = (60,  210, 60)
C_CORRIDOR  = (200, 230, 80)
C_CENTER    = (255, 255, 255)
C_OBS       = (0,   100, 255)   # orange  (obstacles in/near road)
C_HUD       = (0,   230, 80)
C_STOP_HUD  = (0,   0,   220)
C_DETOUR_HUD= (0,   200, 255)

ROAD_ALPHA      = 0.38
CORRIDOR_ALPHA  = 0.45
OBS_ALPHA       = 0.72


# ─────────────────────────────────────────────────────────────────────────────
# Corridor
# ─────────────────────────────────────────────────────────────────────────────

def compute_corridor(free_mask: np.ndarray, car_width_px: int):
    H, W = free_mask.shape
    centerline = np.full(H, -1, np.int32)
    cor_left   = np.full(H, -1, np.int32)
    cor_right  = np.full(H, -1, np.int32)

    for r in range(H):
        row = free_mask[r]
        if not row.any():
            continue
        pad    = np.r_[False, row, False]
        starts = np.where(~pad[:-1] & pad[1:])[0]
        ends   = np.where( pad[:-1] & ~pad[1:])[0]
        widths = ends - starts
        if not widths.size:
            continue
        valid = widths >= car_width_px
        idx   = int(np.argmax(np.where(valid, widths, 0)) if valid.any()
                    else np.argmax(widths))
        cor_left[r]   = starts[idx]
        cor_right[r]  = ends[idx]
        centerline[r] = (starts[idx] + ends[idx]) // 2

    return centerline, cor_left, cor_right


def smooth_centerline(centerline: np.ndarray, W: int, deg: int = 3) -> np.ndarray:
    valid = np.where(centerline >= 0)[0]
    if len(valid) < max(deg + 1, 6):
        return centerline.copy()
    try:
        coeffs   = np.polyfit(valid, centerline[valid].astype(float), deg)
        smoothed = centerline.copy().astype(float)
        smoothed[valid] = np.clip(np.polyval(coeffs, valid), 0, W - 1)
        out = np.round(smoothed).astype(np.int32)
        out[centerline < 0] = -1
        return out
    except np.linalg.LinAlgError:
        return centerline.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Steering (vision-only, from corridor center)
# ─────────────────────────────────────────────────────────────────────────────

def compute_steering(cor_left: np.ndarray, cor_right: np.ndarray,
                     car_width_px: int, H: int, W: int) -> float:
    cx = W // 2
    passable = np.where(
        (cor_left >= 0) & ((cor_right - cor_left) >= car_width_px)
    )[0]
    if passable.size < 3:
        return 0.0
    idx     = max(0, int(passable.size * LOOKAHEAD_FRAC))
    v_L     = int(passable[idx])
    cor_ctr = (cor_left[v_L] + cor_right[v_L]) // 2
    return (cor_ctr - cx) / max(cx, 1)


def steer_label(steer_norm: float) -> str:
    a = abs(steer_norm)
    if a < 0.05:
        return "STRAIGHT"
    side = "RIGHT" if steer_norm > 0 else "LEFT"
    return f"SLIGHT {side}" if a < 0.20 else side


# ─────────────────────────────────────────────────────────────────────────────
# Obstacle detection (in/near the corridor)
# ─────────────────────────────────────────────────────────────────────────────

def detect_obstacles(seg_map: np.ndarray,
                     cor_left: np.ndarray, cor_right: np.ndarray,
                     car_width_px: int) -> list:
    H, W = seg_map.shape
    obs_mask = np.isin(seg_map, list(OBSTACLE_CLASSES)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(obs_mask, 8)

    result = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_OBS_AREA:
            continue
        ox, oy = stats[i, cv2.CC_STAT_LEFT],  stats[i, cv2.CC_STAT_TOP]
        ow, oh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        obs_bottom = oy + oh

        if obs_bottom < H * (1.0 - WARN_ZONE_FRAC):
            continue

        zone    = "STOP" if obs_bottom >= H * (1.0 - STOP_ZONE_FRAC) else "SLOW"
        cy_row  = min(int(centroids[i][1]), H - 1)

        if cor_left[cy_row] < 0:
            continue
        path_l, path_r = cor_left[cy_row], cor_right[cy_row]
        obs_l,  obs_r  = ox, ox + ow

        if obs_r < path_l - CLAMP_MARGIN or obs_l > path_r + CLAMP_MARGIN:
            continue

        left_room  = max(0, obs_l - path_l)
        right_room = max(0, path_r - obs_r)
        detour = ("left"  if left_room  >= car_width_px else
                  "right" if right_room >= car_width_px else None)

        result.append(dict(label_id=i, zone=zone, detour=detour))

    return result, labels   # return label map too for pixel-level drawing


def navigation_status(obstacles: list, steer_norm: float) -> str:
    stop_obs = [o for o in obstacles if o["zone"] == "STOP"]
    slow_obs = [o for o in obstacles if o["zone"] == "SLOW"]
    if stop_obs:
        d = stop_obs[0]["detour"]
        return ("DETOUR LEFT" if d == "left" else
                "DETOUR RIGHT" if d == "right" else "STOP")
    if slow_obs:
        return "SLOW"
    return steer_label(steer_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame(rgb: np.ndarray, seg_map: np.ndarray,
               centerline: np.ndarray) -> np.ndarray:

    H, W  = rgb.shape[:2]
    out   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    guide_top = int(H * (1.0 - GUIDE_FRAC))   # only show guide below this row

    # ── 1. road overlay ───────────────────────────────────────────────────────
    road_mask = np.isin(seg_map, list(ROAD_CLASSES))
    layer = out.copy();  layer[road_mask] = C_ROAD
    cv2.addWeighted(layer, ROAD_ALPHA, out, 1.0 - ROAD_ALPHA, 0, out)

    # ── 2. single blue target dot at nearest centerline point ─────────────────
    cl_valid  = np.where(centerline >= 0)[0]
    near_rows = cl_valid[cl_valid >= guide_top]
    if near_rows.size > 0:
        far_r = int(near_rows[0])
        cv2.circle(out, (int(centerline[far_r]), far_r), 16, (255, 80, 0), -1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(rgb: np.ndarray, processor, model, device,
                  car_width_frac: float) -> np.ndarray:
    H, W = rgb.shape[:2]
    car_width_px = max(1, int(W * car_width_frac))

    seg_map   = infer_seg(rgb, processor, model, device)
    free_mask = np.isin(seg_map, list(ROAD_CLASSES))

    centerline, cor_left, cor_right = compute_corridor(free_mask, car_width_px)
    centerline = smooth_centerline(centerline, W)

    return draw_frame(rgb, seg_map, centerline)


# ─────────────────────────────────────────────────────────────────────────────
# Image / video runners
# ─────────────────────────────────────────────────────────────────────────────

def run_images(input_path: Path, output_dir: Path, processor, model, device,
               car_width_frac: float):
    files = (sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
             if input_path.is_dir() else [input_path])
    output_dir.mkdir(exist_ok=True)
    for f in files:
        print(f"  {f.name} ...", end=" ", flush=True)
        rgb = np.array(Image.open(f).convert("RGB"))
        out = process_frame(rgb, processor, model, device, car_width_frac)
        dst = output_dir / f.name
        Image.fromarray(out).save(dst)
        print(f"→ {dst}")


def run_video(input_path: Path, output_path: Path, processor, model, device,
              car_width_frac: float):
    cap   = cv2.VideoCapture(str(input_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    print(f"  {W}x{H} @ {fps:.1f}fps, {total} frames → {output_path}")
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = process_frame(rgb, processor, model, device, car_width_frac)
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if (i + 1) % 30 == 0:
            print(f"    frame {i+1}/{total}", flush=True)
    cap.release()
    writer.release()
    print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Autonomous navigation overlay")
    p.add_argument("--input",     required=True,
                   help="Image file, folder, or video")
    p.add_argument("--output",    default=None,
                   help="Output folder (images) or .mp4 (video)")
    p.add_argument("--car-width", type=float, default=CAR_WIDTH_FRAC, dest="car_width",
                   help="Vehicle width as fraction of image width (default 0.18)")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    processor, model, device = load_model()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if inp.suffix.lower() in video_exts:
        out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_nav")
        run_video(inp, out, processor, model, device, args.car_width)
    else:
        out = Path(args.output) if args.output else Path(__file__).parent / "output"
        run_images(inp, out, processor, model, device, args.car_width)


if __name__ == "__main__":
    main()
