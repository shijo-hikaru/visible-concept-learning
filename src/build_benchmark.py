"""
Script for regenerating augmented images based on aug_info.json
Works for both CUB and PET datasets.

Usage:
    # CUB
    python src/build_benchmark.py \
        --aug_info cub_visible_part_bench/aug_info.json \
        --src_root /home/user/Develop/datasets/cub/CUB_200_2011/images \
        --bench_dir cub_visible_part_bench

    # PET
    python src/build_benchmark.py \
        --aug_info pet_visible_concept_bench/aug_info.json \
        --src_root /home/user/Develop/datasets/oxford_pet/oxford-iiit-pet/images \
        --bench_dir pet_visible_concept_bench
"""

import argparse
import json
import os
import shutil
from PIL import Image, ImageDraw


def apply_aug(img, entry):
    t = entry["type"]
    if t == "crop":
        img = img.crop((entry["left"], entry["top"], entry["right"], entry["bottom"]))
    elif t == "mask":
        draw = ImageDraw.Draw(img)
        for m in entry["masks"]:
            fill = tuple(m["fill"]) if "fill" in m else (0, 0, 0)
            draw.rectangle([m["left"], m["top"], m["right"], m["bottom"]], fill=fill)
    elif t == "crop+mask":
        c = entry["crop"]
        img = img.crop((c["left"], c["top"], c["right"], c["bottom"]))
        draw = ImageDraw.Draw(img)
        for m in entry["masks"]:
            fill = tuple(m["fill"]) if "fill" in m else (0, 0, 0)
            draw.rectangle([m["left"], m["top"], m["right"], m["bottom"]], fill=fill)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_info", required=True, help="aug_info.json のパス")
    parser.add_argument("--src_root", required=True, help="元画像のルートディレクトリ")
    parser.add_argument("--bench_dir", required=True, help="ベンチマークディレクトリ（images/ と images_re/ を含む）")
    args = parser.parse_args()

    with open(args.aug_info) as f:
        aug_info = json.load(f)

    out_base = os.path.join(args.bench_dir, "images")
    if os.path.exists(out_base):
        shutil.rmtree(out_base)

    errors = []
    for rel, entry in aug_info.items():
        src_path = os.path.join(args.src_root, entry["original"])
        out_path = os.path.join(out_base, entry["output"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if not os.path.exists(src_path):
            errors.append(f"NOT FOUND: {src_path}")
            continue

        img = Image.open(src_path).convert("RGB")
        img = apply_aug(img, entry)
        img.save(out_path)

    print(f"Done. {len(aug_info) - len(errors)} images saved to {out_base}")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
