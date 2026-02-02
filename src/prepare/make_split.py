"""
Generate train/val/test split manifest without shuffling (ordered by video number).
Default ratio: 0.6 / 0.1 / 0.1 for 80 videos.
Output: data/split_manifest.json
"""

import json
from pathlib import Path
import argparse


def make_split(total_videos: int = 80, train_ratio: float = 0.75, val_ratio: float = 0.125, test_ratio: float = 0.125):
    

    video_ids = [f"video{idx:02d}" for idx in range(1, total_videos + 1)]
    n_train = int(total_videos * train_ratio)
    n_val = int(total_videos * val_ratio)

    train = video_ids[:n_train]
    val = video_ids[n_train:n_train + n_val]
    test = video_ids[n_train + n_val:]

    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser(description="Create split manifest without shuffling")
    parser.add_argument('--total_videos', type=int, default=80)
    parser.add_argument('--train_ratio', type=float, default=0.75)
    parser.add_argument('--val_ratio', type=float, default=0.125)
    parser.add_argument('--test_ratio', type=float, default=0.125)
    parser.add_argument('--output', type=str, default='data/split_manifest.json')
    args = parser.parse_args()

    manifest = make_split(args.total_videos, args.train_ratio, args.val_ratio, args.test_ratio)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved split manifest to {out_path}:")
    for k, v in manifest.items():
        print(f"  {k}: {len(v)} videos ({v[0]} -> {v[-1]})")


if __name__ == '__main__':
    main()
