import argparse
import json
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


def build_category_map(coco: Dict) -> Dict[int, str]:
    categories = coco.get("categories", [])
    return {int(cat["id"]): str(cat.get("name", cat.get("id"))) for cat in categories}


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


def poly_to_points(poly: List[float]) -> List[List[float]]:
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    ap.add_argument("--out_dir", required=True, help="Output directory for LabelMe JSONs")
    ap.add_argument("--img_dir", default="", help="Image directory (required if --copy_images)")
    ap.add_argument("--copy_images", action="store_true", help="Copy images to out_dir/images")
    ap.add_argument("--include_crowd", action="store_true", help="Include iscrowd=1 annotations")
    ap.add_argument("--min_area", type=float, default=1.0, help="Min polygon area to keep")
    args = ap.parse_args()

    coco_path = Path(args.coco_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    if args.copy_images:
        if not args.img_dir:
            raise ValueError("--img_dir is required when --copy_images is set")
        images_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(coco_path)
    id_to_name = build_category_map(coco)

    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann.get("image_id", -1))
        anns_by_image.setdefault(img_id, []).append(ann)

    for img_id, img_info in tqdm(images.items(), total=len(images), desc="Images", unit="img"):
        file_name = img_info.get("file_name", "")
        width = int(img_info.get("width", 0))
        height = int(img_info.get("height", 0))
        if not file_name or width <= 0 or height <= 0:
            continue

        shapes: List[Dict] = []
        for ann in anns_by_image.get(img_id, []):
            if not args.include_crowd and int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = int(ann.get("category_id", -1))
            label = id_to_name.get(cat_id, str(cat_id))
            segmentation = ann.get("segmentation")
            polygons = polygons_from_segmentation(segmentation)
            for poly in polygons:
                if polygon_area(poly) < args.min_area:
                    continue
                points = poly_to_points(poly)
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
                shapes.append(shape)

        image_path = file_name
        if args.copy_images and args.img_dir:
            src = Path(args.img_dir) / file_name
            if src.exists():
                dst = images_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                image_path = str(Path("images") / src.name)

        labelme = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        out_path = out_dir / f"{Path(file_name).stem}.json"
        out_path.write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
