#!/usr/bin/env python
# feature_extraction.py
# ------------------------------------------------------------
# Batch extract DINOv2 small/base/large/giant CLS+patch tokens
# and merge save as .npy: (N, 1+P, C) or save only CLS token (N, C)
# ------------------------------------------------------------
# Usage examples:
#
# Cholec80 dataset:
# 1. Process all videos:
#    python feature_extraction.py --dataset cholec80 --models base
#
# 2. Process specified videos:
#    python feature_extraction.py --dataset cholec80 --models base --video_ids video01 video02 video03
#
# 3. Use multiple models:
#    python feature_extraction.py --dataset cholec80 --models base large giant
#
# 4. Custom paths:
#    python feature_extraction.py --dataset cholec80 --data_root ./cholec80 --output_dir ./data/features
# ------------------------------------------------------------
import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
import re
import glob


# ---------- Utility functions ----------
def load_img(p: Path):
    return Image.open(p).convert("RGB")


def batch_tensor(imgs, processor, device):
    return processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)


def save_tokens(out_path: Path, data):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        # Save as .npz file for dictionary data
        np.savez(out_path.with_suffix(".npz"),
                 tokens=data['tokens'].numpy(),
                 frame_ids=data['frame_ids'].numpy())
    else:
        # Save as .npy file for tensor data
        np.save(out_path.with_suffix(".npy"), data.numpy())

def extract_number(file_name):
    match = re.search(r'(\d+)', file_name.split('/')[-1])
    return int(match.group()) if match else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='./cholec80', help='Root directory of the dataset')
    ap.add_argument('--output_dir', type=str, default='./data/features', help='Output directory for features')
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--models", nargs="+",
                    default=["base"],
                    help="choose dinov2 variant: base large giant")
    ap.add_argument("--cpu_fallback", default=True, action="store_true",
                    help="fallback to CPU when GPU memory is insufficient")
    ap.add_argument("--video_ids", nargs="+", default=None,
                    help="specify video IDs, e.g., video01 video02; defaults to all")
    ap.add_argument('--split_manifest', type=str, default=None,
                    help='Path to split_manifest.json; if provided, only process listed videos and place outputs under train/val/test subfolders')
    args = ap.parse_args()

    vids = [] #dir paths of each video
    vid_names = [] #video names
    vid_split = {}  # video_name -> split
    allowed = None
    if args.split_manifest:
        with open(args.split_manifest, 'r') as f:
            manifest = json.load(f)
        allowed = set(manifest.get('train', []) + manifest.get('val', []) + manifest.get('test', []))
        for name in manifest.get('train', []):
            vid_split[name] = 'train'
        for name in manifest.get('val', []):
            vid_split[name] = 'val'
        for name in manifest.get('test', []):
            vid_split[name] = 'test'

    root = args.data_root
    frames_dir = os.path.join(root, "frames")

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    target_list = args.video_ids if args.video_ids else sorted([item for item in os.listdir(frames_dir) if item.startswith('video') and not item.startswith('.')])
    for vid in target_list:
        if allowed is not None and vid not in allowed:
            continue
        vid_path = os.path.join(frames_dir, vid)
        if os.path.isdir(vid_path):
            vids.append(vid_path)
            vid_names.append(vid)
        else:
            print(f"[Warning] Video directory not found: {vid_path}")

    print(f"[Info] Found {len(vids)} videos to process: {vid_names[:5]}{'...' if len(vid_names) > 5 else ''}")

    output_dir = args.output_dir

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_map = {
        "base":  "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for m in args.models:
        print(f"\n=== {id_map[m]} ===")
        # Load model
        processor = AutoImageProcessor.from_pretrained(id_map[m], size={"height": 224, "width": 224}, do_center_crop=False)
        try:
            model = AutoModel.from_pretrained(id_map[m]).to(device)
        except RuntimeError as e:
            if args.cpu_fallback:
                print(f"[Warn] Insufficient GPU memory, switching to CPU: {e}")
                device = torch.device("cpu")
                model = AutoModel.from_pretrained(id_map[m]).to(device)
            else:
                raise
        model.eval()

        for i in range(len(vids)):
            vid_name = vid_names[i]

            # Get all images according to dataset
            # Cholec80: frames stored as videoXX_000001.png directly under the video folder
            frames = glob.glob(os.path.join(vids[i], "*.png"))

            # sort the frames
            frames = sorted(frames, key=extract_number)
            N = len(frames)

            if N == 0:
                print(f"[Warning] No frames found in {vids[i]}")
                continue

            token_list = []  # Use list to collect tensors
            frame_id_list = []  # Collect frame IDs

            for s in tqdm(range(0, N, args.batch_size), desc=id_map[m]):
                paths = frames[s: s + args.batch_size]
                imgs  = [load_img(p) for p in paths]
                x     = batch_tensor(imgs, processor, device)

                # Extract frame IDs for this batch
                batch_frame_ids = [extract_number(p) for p in paths]
                frame_id_list.extend(batch_frame_ids)

                with torch.no_grad():
                    tokens = model(pixel_values=x).last_hidden_state.float().cpu()  # (B, 1+P, C)
                    tokens = tokens[:, 0, :]  # Keep only the CLS token (C)

                token_list.append(tokens)

            # Concatenate all tokens into a single tensor
            token_tensor = torch.cat(token_list, dim=0)  # (N, C)
            frame_ids = torch.tensor(frame_id_list)  # (N,)

            # Save both tokens and frame IDs
            data_dict = {
                'tokens': token_tensor,
                'frame_ids': frame_ids
            }
            tag = id_map[m].split("/")[-1] + '-cls'             
            split_subdir = vid_split.get(vid_name, None)
            save_path = out_dir / tag
            if split_subdir:
                save_path = save_path / split_subdir
            save_tokens(save_path / f"{vid_name}", data_dict)
            print(f"[Saved] tokens: {token_tensor.shape}, frame_ids: {frame_ids.shape} →  {save_path / (vid_name + '.npz')}")

    print(f"\n[Done] {args.dataset} {args.models} tokens extraction finished.")


if __name__ == "__main__":
    main()