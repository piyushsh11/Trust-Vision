"""
CI guardrail: run a small robustness sweep on a curated image folder.

Usage:
  python robustness_check.py --data-dir images --dataset items --attack pgd --epsilon 8 --threshold 0.6
Exit code 1 when robust accuracy drops below threshold.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from backend import (
    auto_attack,
    cw_attack,
    pgd_attack,
    predict,
    preprocess_no_norm,
)


def main():
    parser = argparse.ArgumentParser(description="Robustness guardrail check")
parser.add_argument("--data-dir", default="images", help="Folder with evaluation images")
parser.add_argument("--dataset", default="set1", choices=["set1", "items", "medmnist"], help="Dataset head to use for labels")
parser.add_argument("--attack", default="pgd", choices=["pgd", "auto", "cw"], help="Attack to run")
parser.add_argument("--epsilon", type=float, default=8, help="Epsilon in 1/255 units")
parser.add_argument("--threshold", type=float, default=0.6, help="Minimum robust accuracy required")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    imgs = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not imgs:
        print(f"No images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    eps = args.epsilon / 255.0
    attack_fn = {"pgd": pgd_attack, "auto": auto_attack, "cw": cw_attack}[args.attack]

    clean = 0
    robust = 0

    for path in imgs:
        img = Image.open(path).convert("RGB")
        label, _ = predict(img, dataset=args.dataset)
        clean += 1

        adv_img = attack_fn(img, eps)
        adv_label, _ = predict(adv_img, dataset=args.dataset)
        if adv_label == label:
            robust += 1

    clean_acc = clean / len(imgs)
    robust_acc = robust / len(imgs)

    print(f"Images evaluated: {len(imgs)}")
    print(f"Clean accuracy (proxy): {clean_acc:.2f}")
    print(f"Robust accuracy ({args.attack} @ {args.epsilon}/255): {robust_acc:.2f}")
    if robust_acc < args.threshold:
        print(f"FAIL: robust_acc {robust_acc:.2f} < threshold {args.threshold}", file=sys.stderr)
        sys.exit(1)

    print("PASS: robustness above threshold")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
