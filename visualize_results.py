import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(coco: Dict) -> Dict[int, str]:
    categories = coco.get("categories", [])
    return {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in categories}


def color_for_category(category_id: int) -> Tuple[int, int, int]:
    # Deterministic color based on category id
    rng = np.random.default_rng(category_id * 9973)
    color = rng.integers(30, 255, size=3, dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def decode_rle(segmentation: Dict) -> np.ndarray:
    rle = dict(segmentation)
    counts = rle.get("counts")
    if isinstance(counts, str):
        rle["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def draw_label(
    image: np.ndarray,
    label: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x0 = max(0, x)
    y0 = max(0, y - th - 6)
    cv2.rectangle(image, (x0, y0), (x0 + tw + 6, y0 + th + 6), color, -1)
    cv2.putText(
        image,
        label,
        (x0 + 3, y0 + th + 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def visualize_image(
    image_path: Path,
    annotations: List[Dict],
    categories: Dict[int, str],
    alpha: float,
    show_bbox: bool,
) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    overlay = bgr.copy()
    mask_canvas = np.zeros_like(bgr)

    for ann in annotations:
        segmentation = ann.get("segmentation")
        if not isinstance(segmentation, dict):
            continue

        mask = decode_rle(segmentation)
        if mask.shape[0] != bgr.shape[0] or mask.shape[1] != bgr.shape[1]:
            # Skip masks that don't match the image size
            continue

        category_id = int(ann.get("category_id", -1))
        label = categories.get(category_id, str(category_id))
        color = color_for_category(category_id)

        mask_canvas[mask > 0] = color
        overlay[mask > 0] = (
            overlay[mask > 0].astype(np.float32) * (1.0 - alpha)
            + np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)

        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = [int(round(v)) for v in bbox]
            if show_bbox:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            draw_label(overlay, label, x, y, color)


    return overlay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", required=True, help="Path to COCO annotation JSON")
    ap.add_argument("--img_dir", required=True, help="Directory with images")
    ap.add_argument("--max_images", type=int, default=5000, help="Max images to render")
    ap.add_argument("--alpha", type=float, default=0.55, help="Mask overlay alpha")
    ap.add_argument("--show_bbox", action="store_true", help="Draw bounding boxes")
    ap.add_argument("--max_width", type=int, default=1280, help="Max window width")
    ap.add_argument("--max_height", type=int, default=720, help="Max window height")
    args = ap.parse_args()

    coco_path = Path(args.coco_json)
    img_dir = Path(args.img_dir)

    coco = load_coco(coco_path)
    categories = build_category_map(coco)

    ann_by_image: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = int(ann.get("image_id", -1))
        ann_by_image.setdefault(img_id, []).append(ann)

    images = coco.get("images", [])
    for idx, img_info in enumerate(images[: args.max_images]):
        img_id = int(img_info.get("id", -1))
        file_name = img_info.get("file_name", "")
        img_path = img_dir / file_name
        if not img_path.exists():
            continue

        try:
            result = visualize_image(
                img_path,
                ann_by_image.get(img_id, []),
                categories,
                alpha=args.alpha,
                show_bbox=args.show_bbox
            )
        except Exception:
            continue

        window_title = f"{file_name} ({idx + 1}/{min(len(images), args.max_images)})"
        view = result
        h, w = view.shape[:2]
        scale = min(args.max_width / w, args.max_height / h, 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            view = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, view.shape[1], view.shape[0])
        cv2.imshow(window_title, view)
        key = cv2.waitKey(0)
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
