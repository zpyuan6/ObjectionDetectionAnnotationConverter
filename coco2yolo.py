import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(coco: Dict) -> Tuple[Dict[int, int], List[str]]:
    categories = coco.get("categories", [])
    categories = sorted(categories, key=lambda c: int(c.get("id", 0)))
    id_to_index = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [str(cat.get("name", cat.get("id"))) for cat in categories]
    return id_to_index, names


def decode_rle(segmentation: Dict) -> np.ndarray:
    rle = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def polygon_area(poly: List[float]) -> float:
    if len(poly) < 6:
        return 0.0
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return float(cv2.contourArea(pts))


def polygons_from_segmentation(segmentation: object) -> List[List[float]]:
    if isinstance(segmentation, list):
        polys = []
        for poly in segmentation:
            if isinstance(poly, list) and len(poly) >= 6:
                polys.append(poly)
        return polys
    if isinstance(segmentation, dict):
        mask = decode_rle(segmentation)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            poly = cnt.reshape(-1, 2).astype(np.float32).flatten().tolist()
            if len(poly) >= 6:
                polys.append(poly)
        return polys
    return []


def normalize_polygon(poly: List[float], width: int, height: int) -> List[float]:
    coords = []
    for idx, val in enumerate(poly):
        if idx % 2 == 0:
            coords.append(val / width)
        else:
            coords.append(val / height)
    return coords


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    ap.add_argument("--out_dir", required=True, help="Output directory for YOLO labels")
    ap.add_argument("--img_dir", default="", help="Image directory (required if --copy_images)")
    ap.add_argument("--copy_images", action="store_true", help="Copy images to out_dir/images")
    ap.add_argument("--split", action="store_true", help="Split dataset into train/val/test")
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio")
    ap.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--keep_multi", action="store_true", help="Write all polygons per object")
    ap.add_argument("--min_area", type=float, default=1.0, help="Min polygon area to keep")
    args = ap.parse_args()

    coco_path = Path(args.coco_json)
    out_dir = Path(args.out_dir)
    labels_dir = out_dir / "labels"
    images_dir = out_dir / "images"
    if args.copy_images:
        if not args.img_dir:
            raise ValueError("--img_dir is required when --copy_images is set")
        images_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
        if ratio_sum <= 0 or ratio_sum > 1.0 + 1e-6:
            raise ValueError("Split ratios must sum to > 0 and <= 1.0")

    coco = load_coco(coco_path)
    id_to_index, class_names = build_category_map(coco)

    classes_path = out_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names), encoding="utf-8")

    images = {int(img["id"]): img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])

    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in annotations:
        img_id = int(ann.get("image_id", -1))
        anns_by_image.setdefault(img_id, []).append(ann)

    valid_image_ids = []
    for img_id, img_info in images.items():
        file_name = img_info.get("file_name", "")
        width = int(img_info.get("width", 0))
        height = int(img_info.get("height", 0))
        if file_name and width > 0 and height > 0:
            valid_image_ids.append(img_id)

    split_map: Dict[int, str] = {}
    if args.split and valid_image_ids:
        rng = random.Random(args.seed)
        rng.shuffle(valid_image_ids)
        total = len(valid_image_ids)
        n_train = int(total * args.train_ratio)
        n_val = int(total * args.val_ratio)
        n_test = total - n_train - n_val
        train_ids = set(valid_image_ids[:n_train])
        val_ids = set(valid_image_ids[n_train:n_train + n_val])
        test_ids = set(valid_image_ids[n_train + n_val:])
        for img_id in train_ids:
            split_map[img_id] = "train"
        for img_id in val_ids:
            split_map[img_id] = "val"
        for img_id in test_ids:
            split_map[img_id] = "test"

        for split in ("train", "val", "test"):
            (labels_dir / split).mkdir(parents=True, exist_ok=True)
            if args.copy_images:
                (images_dir / split).mkdir(parents=True, exist_ok=True)
    else:
        labels_dir.mkdir(parents=True, exist_ok=True)

    split_lists: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    def image_ref(file_name: str, split: str) -> str:
        if args.copy_images:
            return str(Path("images") / split / file_name)
        if args.img_dir:
            return str(Path(args.img_dir) / file_name)
        return file_name

    for img_id, img_info in tqdm(images.items(), total=len(images), desc="Images", unit="img"):
        file_name = img_info.get("file_name", "")
        width = int(img_info.get("width", 0))
        height = int(img_info.get("height", 0))
        if not file_name or width <= 0 or height <= 0:
            continue

        split = split_map.get(img_id, "train") if args.split else ""
        label_dir = labels_dir / split if args.split else labels_dir

        lines: List[str] = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in id_to_index:
                continue
            segmentation = ann.get("segmentation")
            polygons = polygons_from_segmentation(segmentation)
            if not polygons:
                continue

            if not args.keep_multi:
                polygons = sorted(polygons, key=polygon_area, reverse=True)[:1]

            for poly in polygons:
                if polygon_area(poly) < args.min_area:
                    continue
                norm = normalize_polygon(poly, width, height)
                norm = [min(1.0, max(0.0, v)) for v in norm]
                line = " ".join([str(id_to_index[cat_id])] + [f"{v:.6f}" for v in norm])
                lines.append(line)

        label_path = label_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")

        if args.copy_images and args.img_dir:
            src = Path(args.img_dir) / file_name
            if src.exists():
                dst = images_dir / split / src.name if args.split else images_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        if args.split:
            split_lists[split].append(image_ref(file_name, split))

    if args.split:
        for split, items in split_lists.items():
            (out_dir / f"{split}.txt").write_text("\n".join(items), encoding="utf-8")


if __name__ == "__main__":
    main()
