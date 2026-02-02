#!/usr/bin/env python
# feature_extraction.py
# ------------------------------------------------------------
# 批量提取 DINOv2 small/base/large/giant 的 CLS+patch token
# 并合并保存为 .npy：(N, 1+P, C) 或只保存 CLS token (N, C)
# ------------------------------------------------------------
# 使用示例:
# 
# Cholec80 数据集:
# 1. 处理所有视频:
#    python feature_extraction.py --dataset cholec80 --models base
# 
# 2. 处理指定视频:
#    python feature_extraction.py --dataset cholec80 --models base --video_ids video01 video02 video03
#
# 3. 使用多个模型:
#    python feature_extraction.py --dataset cholec80 --models base large giant
#
# 4. 自定义路径:
#    python feature_extraction.py --dataset cholec80 --data_root ./cholec80 --output_dir ./data/features
# ------------------------------------------------------------
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel
import re
import glob


# ---------- 工具函数 ----------
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
    ap.add_argument('--dataset', type=str, choices=['rarp50', 'vvs', 'tme', 'cholec80'], default='cholec80', help='Dataset to use for feature extraction')
    ap.add_argument('--data_root', type=str, default='./cholec80', help='Root directory of the dataset')
    ap.add_argument('--output_dir', type=str, default='./data/features', help='Output directory for features')
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--models", nargs="+",
                    default=["base"],
                    help="选择 dinov2 版本: base large giant")
    ap.add_argument("--cpu_fallback", default=True, action="store_true",
                    help="显存不足时是否转用 CPU 进行推断")
    ap.add_argument("--video_ids", nargs="+", default=None,
                    help="指定处理的视频ID，如 video01 video02，默认处理所有")
    args = ap.parse_args()

    vids = [] #dir paths of each video
    vid_names = [] #video names
    
    if args.dataset == "cholec80":
        # Cholec80 数据集路径结构
        root = args.data_root
        frames_dir = os.path.join(root, "frames")
        
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        # 获取所有视频文件夹
        if args.video_ids:
            # 处理指定的视频
            for vid in args.video_ids:
                vid_path = os.path.join(frames_dir, vid)
                if os.path.isdir(vid_path):
                    vids.append(vid_path)
                    vid_names.append(vid)
                else:
                    print(f"[Warning] Video directory not found: {vid_path}")
        else:
            # 处理所有视频（video01-video80）
            for item in sorted(os.listdir(frames_dir)):
                if item.startswith('video') and not item.startswith('.'):
                    item_path = os.path.join(frames_dir, item)
                    if os.path.isdir(item_path):
                        vids.append(item_path)
                        vid_names.append(item)
        
        print(f"[Info] Found {len(vids)} videos to process: {vid_names[:5]}{'...' if len(vid_names) > 5 else ''}")
        
    elif args.dataset == "rarp50":
        root = "/home/rmapchb/Reserch/data/rarp50_data/"
        for category in ['train', 'test']:
            category_path = os.path.join(root, category)
            # Check if the directory exists
            if os.path.exists(category_path):
                for item in os.listdir(category_path):
                    item_path = os.path.join(category_path, item)
                    if os.path.isdir(item_path):
                        vids.append(item_path)
                        vid_names.append(item)
    elif args.dataset == "vvs":
        root = "/home/rmapchb/Reserch/data/raw/frames/"
        for item in os.listdir(root):
            # 只处理文件夹
            if item.startswith('.'):
                continue
            else:
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path):
                    vids.append(item_path)
                    vid_names.append(item)
    elif args.dataset == "tme":
        root = "/media/HDD1/jialang/Error_Detection/15TME_video_select/"
        for item in os.listdir(root):
            # 只处理文件夹
            if item.startswith('.'):
                continue
            else:
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path):
                    vids.append(item_path)
                    vid_names.append(item)
    
    # 设置输出目录
    if args.dataset == "cholec80":
        output_dir = args.output_dir
    elif args.dataset == "rarp50":
        output_dir = "/media/HDD1/jialang/CMLSED/Data/rarp50_token"
    elif args.dataset == "vvs":
        output_dir = "/home/rmapchb/Reserch/data/raw/"
    elif args.dataset == "tme":
        output_dir = "/media/HDD1/jialang/CMLSED/Data/tme_token"
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_map = {
        # "small": "facebook/dinov2-small",
        "base":  "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for m in args.models:
        print(f"\n=== {id_map[m]} ===")
        # 载入模型
        processor = AutoImageProcessor.from_pretrained(id_map[m], size={"height": 224, "width": 224}, do_center_crop=False)
        try:
            model = AutoModel.from_pretrained(id_map[m]).to(device)
        except RuntimeError as e:
            if args.cpu_fallback:
                print(f"[Warn] 显存不足，转用 CPU: {e}")
                device = torch.device("cpu")
                model = AutoModel.from_pretrained(id_map[m]).to(device)
            else:
                raise
        model.eval()

        for i in range(len(vids)):
            vid_name = vid_names[i]
            
            # 根据数据集获取所有图片
            if args.dataset == "cholec80":
                # Cholec80: 直接在视频文件夹下，文件名格式 video01_000001.png
                frames = glob.glob(os.path.join(vids[i], "*.png"))
            elif args.dataset == "rarp50":
                frames = glob.glob(vids[i]+"/frame_10HZ/" + r"/*.png")
            elif args.dataset == "vvs":
                frames = glob.glob(vids[i]+"/frame_10HZ/" + r"/*.png")
            elif args.dataset == "tme":
                frames = glob.glob(vids[i]+"/frame_10HZ/" + r"/*.jpg")
            
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
            tag = id_map[m].split("/")[-1] + '-cls'              # dinov2-small, dinov2-base ...
            save_tokens(out_dir / tag / f"{vid_name}", data_dict)
            print(f"[Saved] tokens: {token_tensor.shape}, frame_ids: {frame_ids.shape} →  {out_dir / tag / f'{vid_name}.npy'}")

    print(f"\n[Done] {args.dataset} {args.models} tokens extraction finished.")


if __name__ == "__main__":
    main()