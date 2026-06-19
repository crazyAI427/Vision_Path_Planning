#!/usr/bin/env python
"""Drivable-region segmentation using SegFormer-B4 (Cityscapes).

Usage
-----
# all PNGs/JPGs in a folder
python inference.py --input input/

# single image
python inference.py --input input/1.png

# video
python inference.py --input test.mp4 --output test_out.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

MODEL_ID = "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
MODEL_DIR = Path(__file__).parent / "model"

DRIVABLE_CLASS = 0          # Cityscapes trainId 0 = road
OVERLAY_COLOR = (0, 200, 80)
OVERLAY_ALPHA = 0.45


# ── model ─────────────────────────────────────────────────────────────────────

def load_model():
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, cache_dir=MODEL_DIR)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Running on: {device}")
    return processor, model, device


# ── core inference ─────────────────────────────────────────────────────────────

def infer_mask(rgb_np: np.ndarray, processor, model, device) -> np.ndarray:
    """Return a boolean (H, W) drivable-region mask for an RGB numpy array."""
    image = Image.fromarray(rgb_np)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, C, H/4, W/4)
    logits = F.interpolate(logits, size=(image.height, image.width),
                           mode="bilinear", align_corners=False)
    seg = logits.argmax(dim=1)[0].cpu().numpy()

    mask = (seg == DRIVABLE_CLASS).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def apply_overlay(rgb_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb_np.copy()
    layer = out.copy()
    layer[mask] = OVERLAY_COLOR
    return (out * (1 - OVERLAY_ALPHA) + layer * OVERLAY_ALPHA).astype(np.uint8)


# ── image mode ────────────────────────────────────────────────────────────────

def run_images(input_path: Path, output_dir: Path, processor, model, device):
    if input_path.is_dir():
        files = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        files = [input_path]

    output_dir.mkdir(exist_ok=True)
    for f in files:
        print(f"  {f.name} ...", end=" ", flush=True)
        rgb = np.array(Image.open(f).convert("RGB"))
        mask = infer_mask(rgb, processor, model, device)
        result = apply_overlay(rgb, mask)
        out = output_dir / f.name
        Image.fromarray(result).save(out)
        print(f"→ {out}")


# ── video mode ────────────────────────────────────────────────────────────────

def run_video(input_path: Path, output_path: Path, processor, model, device):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"  {w}x{h} @ {fps:.1f}fps, {total} frames → {output_path}")
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = infer_mask(rgb, processor, model, device)
        result = apply_overlay(rgb, mask)
        writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if (i + 1) % 30 == 0:
            print(f"    frame {i+1}/{total}", flush=True)

    cap.release()
    writer.release()
    print("  Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drivable-region segmentation")
    parser.add_argument("--input", required=True,
                        help="Image file, image folder, or video file (.mp4/.avi/...)")
    parser.add_argument("--output",
                        help="Output path. For images: output folder (default: ./output). "
                             "For video: output .mp4 path (default: <input>_out.mp4)")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    processor, model, device = load_model()

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if inp.suffix.lower() in video_exts:
        out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_out")
        run_video(inp, out, processor, model, device)
    else:
        out = Path(args.output) if args.output else Path(__file__).parent / "output"
        run_images(inp, out, processor, model, device)


if __name__ == "__main__":
    main()
