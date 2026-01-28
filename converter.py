import json
import argparse
import contextlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

# SAM2 via Transformers (Hugging Face)
from transformers import Sam2Model, Sam2Processor


def parse_voc_xml(xml_path: Path) -> Tuple[str, int, int, List[Dict]]:
    """
    Returns: (filename, width, height, objects)
    objects: [{"name": str, "bbox_xyxy": [xmin,ymin,xmax,ymax]}]
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # check if filename in XML matches the actual file name
    real_file_name = xml_path.stem
    filename = root.findtext("filename")
    # If filename has an extension, strip it for comparison
    file_path = Path(filename)
    file_stem = file_path.stem
    if file_stem != real_file_name:
        # Update XML file to match actual file name with extension
        print(f"[INFO] Updating filename in XML from {filename} to match {xml_path.name}")
        file_extension = file_path.suffix
        filename = real_file_name + file_extension
        root.find("filename").text = filename
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        b = obj.find("bndbox")
        xmin = int(float(b.findtext("xmin")))
        ymin = int(float(b.findtext("ymin")))
        xmax = int(float(b.findtext("xmax")))
        ymax = int(float(b.findtext("ymax")))
        objects.append({"name": name, "bbox_xyxy": [xmin, ymin, xmax, ymax]})

    return filename, width, height, objects


def coco_rle_from_mask(binary_mask: np.ndarray) -> Dict:
    """
    COCO expects RLE with counts as a JSON-serializable string (utf-8).
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    # pycocotools expects Fortran order
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def bbox_xywh_from_mask(binary_mask: np.ndarray) -> List[float]:
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def area_from_rle(rle: Dict) -> float:
    # Need bytes counts for pycocotools area
    rle_bytes = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
    return float(mask_utils.area(rle_bytes))


def main():
    def info(msg: str):
        tqdm.write(f"[INFO] {msg}")

    def warn(msg: str):
        tqdm.write(f"[WARN] {msg}")

    def load_existing_coco(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True, help="Directory with Pascal VOC XML files")
    ap.add_argument(
        "--xml_file",
        default="",
        help="Optional XML filename or path to update its annotations in an existing COCO json",
    )
    ap.add_argument("--img_dir", required=True, help="Directory with corresponding images")
    ap.add_argument("--out_json", required=True, help="Output COCO json path")
    ap.add_argument(
        "--sam2_model_id",
        default="facebook/sam2-hiera-large",
        help="Hugging Face model id (e.g., facebook/sam2-hiera-large)",
    )
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--clip_to_box", action="store_true",
                    help="Intersect predicted mask with the input bbox to reduce spillover")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.img_dir)
    out_path = Path(args.out_json)

    # Load SAM2 from Hugging Face (Transformers)
    info(f"Loading SAM2 model from Hugging Face: {args.sam2_model_id}, device={args.device}")
    processor = Sam2Processor.from_pretrained(args.sam2_model_id)
    model = Sam2Model.from_pretrained(args.sam2_model_id).to(device=args.device)
    model.eval()
    info("SAM2 model loaded successfully.")

    update_existing = False
    category_name_to_id: Dict[str, int] = {}
    categories: List[Dict] = []
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    img_id = 1

    if out_path.exists() and args.xml_file:
        existing = load_existing_coco(out_path)
        categories = list(existing.get("categories", []))
        images = list(existing.get("images", []))
        annotations = list(existing.get("annotations", []))
        category_name_to_id = {c["name"]: int(c["id"]) for c in categories if "id" in c and "name" in c}
        ann_id = max([int(a.get("id", 0)) for a in annotations], default=0) + 1
        img_id = max([int(i.get("id", 0)) for i in images], default=0) + 1
        update_existing = True

    if args.xml_file:
        xml_path = Path(args.xml_file)
        if not xml_path.is_absolute():
            xml_path = xml_dir / xml_path
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        xml_paths = [xml_path]
    else:
        xml_paths = sorted(xml_dir.glob("*.xml"))

    if not xml_paths:
        raise FileNotFoundError(f"No .xml found in {xml_dir}")
    info(f"Found {len(xml_paths)} XML files.")

    xml_items = []
    total_objects = 0
    for xml_path in xml_paths:
        try:
            filename, w, h, objects = parse_voc_xml(xml_path)
        except Exception as exc:
            warn(f"Failed to parse {xml_path.name}: {exc}. Skipping.")
            continue
        xml_items.append((xml_path, filename, w, h, objects))
        total_objects += len(objects)

    if not xml_items:
        raise FileNotFoundError(f"No valid .xml found in {xml_dir}")

    info(f"Total objects (bboxes) to process: {total_objects}")

    images_bar = tqdm(total=len(xml_items), desc="Images", unit="img")
    ann_bar = tqdm(total=total_objects, desc="Annotations", unit="ann")
    use_autocast = args.device.startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

    try:
        with torch.inference_mode(), autocast_ctx:
            for xml_path, filename, w, h, objects in xml_items:
                img_path = img_dir / filename
                if not img_path.exists():
                    # If filename in XML doesn't exist, try same stem with common extensions
                    stem = Path(filename).stem
                    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG", ".tif", ".tiff"]:
                        cand = img_dir / f"{stem}{ext}"
                        if cand.exists():
                            img_path = cand
                            break

                if not img_path.exists():
                    warn(f"Missing image for {xml_path.name}: expected {filename}. Skipping.")
                    images_bar.update(1)
                    continue

                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    warn(f"Failed to read image: {img_path}. Skipping.")
                    images_bar.update(1)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                current_img_id = None
                if update_existing:
                    existing_img = next((img for img in images if img.get("file_name") == img_path.name), None)
                    if existing_img is not None:
                        current_img_id = int(existing_img.get("id", img_id))
                        images = [img for img in images if img.get("id") != current_img_id]
                        annotations = [ann for ann in annotations if int(ann.get("image_id", -1)) != current_img_id]

                if current_img_id is None:
                    current_img_id = img_id
                    img_id += 1

                images.append({
                    "id": current_img_id,
                    "file_name": img_path.name,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                })

                input_boxes = []
                for obj in objects:
                    xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
                    input_boxes.append([xmin, ymin, xmax, ymax])

                inputs = processor(
                    images=rgb,
                    input_boxes=[input_boxes],
                    return_tensors="pt",
                ).to(args.device)

                outputs = model(**inputs, multimask_output=False)
                processed_masks = processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"],
                )

                masks_for_image = processed_masks[0] if isinstance(processed_masks, list) else processed_masks[0]

                if masks_for_image.ndim == 4:
                    # (num_boxes, num_masks, H, W) -> select best mask per box
                    if masks_for_image.shape[1] == 1:
                        masks_for_image = masks_for_image[:, 0, :, :]
                    else:
                        scores = getattr(outputs, "iou_scores", None)
                        if scores is not None:
                            scores = scores[0].cpu().numpy()
                            best_idx = np.argmax(scores, axis=1)
                            selected = []
                            for box_idx, mask_idx in enumerate(best_idx):
                                selected.append(masks_for_image[box_idx, mask_idx])
                            masks_for_image = np.stack(selected, axis=0)
                        else:
                            masks_for_image = masks_for_image[:, 0, :, :]
                elif masks_for_image.ndim != 3:
                    warn(f"Unexpected mask shape: {tuple(masks_for_image.shape)} for {img_path.name}")

                num_masks = masks_for_image.shape[0] if hasattr(masks_for_image, "shape") else 0
                if num_masks != len(objects):
                    warn(
                        f"Mask count mismatch for {img_path.name}: "
                        f"{num_masks} masks for {len(objects)} boxes."
                    )

                for idx, obj in enumerate(objects[:num_masks]):
                    name = obj["name"]
                    if name not in category_name_to_id:
                        cid = max(category_name_to_id.values(), default=0) + 1
                        category_name_to_id[name] = cid
                        categories.append({"id": cid, "name": name, "supercategory": "object"})

                    xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
                    mask = masks_for_image[idx]
                    if torch.is_tensor(mask):
                        mask = mask.numpy()
                    mask = (mask > 0).astype(np.uint8)

                    if args.clip_to_box:
                        clipped = np.zeros_like(mask, dtype=np.uint8)
                        x0, y0, x1, y1 = map(int, [xmin, ymin, xmax, ymax])
                        x0 = max(0, x0); y0 = max(0, y0)
                        x1 = min(mask.shape[1] - 1, x1); y1 = min(mask.shape[0] - 1, y1)
                        clipped[y0:y1+1, x0:x1+1] = mask[y0:y1+1, x0:x1+1]
                        mask = clipped

                    rle = coco_rle_from_mask(mask)
                    bbox_xywh = bbox_xywh_from_mask(mask)
                    area = area_from_rle(rle)

                    annotations.append({
                        "id": ann_id,
                        "image_id": current_img_id,
                        "category_id": category_name_to_id[name],
                        "segmentation": rle,   # COCO RLE
                        "bbox": bbox_xywh,
                        "area": area,
                        "iscrowd": 0,
                    })
                    ann_id += 1
                    ann_bar.update(1)

                images_bar.update(1)
    finally:
        images_bar.close()
        ann_bar.close()

    coco = {
        "info": {"description": "Converted from Pascal VOC (XML) boxes to COCO instance masks using SAM2"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    info(f"Saved COCO JSON: {out_path} (images={len(images)}, anns={len(annotations)})")


if __name__ == "__main__":
    main()
