#!/usr/bin/env python
"""Autonomous path planning overlay.

Visual output:
  Green  overlay = drivable road region
  Blue   dot     = target waypoint (farthest point of near-field corridor centre)

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

ROAD_CLASSES   = {0, 9}      # Cityscapes trainId: road, terrain
CAR_WIDTH_FRAC = 0.18        # vehicle width as fraction of image width
GUIDE_FRAC     = 0.20        # blue dot placed within bottom GUIDE_FRAC of frame

C_ROAD         = (60, 210, 60)
ROAD_ALPHA     = 0.38
C_TARGET       = (255, 80, 0)   # BGR blue
TARGET_RADIUS  = 16


def compute_centerline(free_mask: np.ndarray, car_width_px: int) -> np.ndarray:
    H, _ = free_mask.shape
    centerline = np.full(H, -1, np.int32)
    for r in range(H):
        row = free_mask[r]
        if not row.any():
            continue
        pad    = np.r_[False, row, False]
        starts = np.where(~pad[:-1] &  pad[1:])[0]
        ends   = np.where( pad[:-1] & ~pad[1:])[0]
        widths = ends - starts
        if not widths.size:
            continue
        valid = widths >= car_width_px
        idx   = int(np.argmax(np.where(valid, widths, 0)) if valid.any()
                    else np.argmax(widths))
        centerline[r] = (starts[idx] + ends[idx]) // 2
    return centerline


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


def draw_frame(rgb: np.ndarray, seg_map: np.ndarray,
               centerline: np.ndarray) -> np.ndarray:
    H, W  = rgb.shape[:2]
    out   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    road_mask = np.isin(seg_map, list(ROAD_CLASSES))
    layer = out.copy()
    layer[road_mask] = C_ROAD
    cv2.addWeighted(layer, ROAD_ALPHA, out, 1.0 - ROAD_ALPHA, 0, out)

    guide_top = int(H * (1.0 - GUIDE_FRAC))
    cl_valid  = np.where(centerline >= 0)[0]
    near_rows = cl_valid[cl_valid >= guide_top]
    if near_rows.size > 0:
        far_r = int(near_rows[0])
        cv2.circle(out, (int(centerline[far_r]), far_r), TARGET_RADIUS, C_TARGET, -1)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def process_frame(rgb: np.ndarray, processor, model, device,
                  car_width_frac: float) -> np.ndarray:
    H, W = rgb.shape[:2]
    car_width_px = max(1, int(W * car_width_frac))

    seg_map    = infer_seg(rgb, processor, model, device)
    free_mask  = np.isin(seg_map, list(ROAD_CLASSES))
    centerline = compute_centerline(free_mask, car_width_px)
    centerline = smooth_centerline(centerline, W)

    return draw_frame(rgb, seg_map, centerline)


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


def main():
    p = argparse.ArgumentParser(description="Autonomous navigation overlay")
    p.add_argument("--input",     required=True, help="Image file, folder, or video")
    p.add_argument("--output",    default=None,  help="Output folder (images) or .mp4 (video)")
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
