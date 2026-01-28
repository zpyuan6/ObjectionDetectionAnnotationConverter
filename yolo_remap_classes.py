import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


def load_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_name_map(old_classes: List[str], new_classes: List[str]) -> Dict[int, int]:
    new_index = {name: idx for idx, name in enumerate(new_classes)}
    mapping: Dict[int, int] = {}
    for idx, name in enumerate(old_classes):
        if name in new_index:
            mapping[idx] = new_index[name]
    return mapping


def load_mapping_json(path: Path) -> Dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[int, int] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                mapping[int(k)] = int(v)
            except Exception:
                continue
    return mapping


def remap_label_file(
    src: Path,
    dst: Path,
    mapping: Dict[int, int],
    drop_missing: bool,
) -> None:
    lines = src.read_text(encoding="utf-8").splitlines()
    out_lines: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        if old_id not in mapping:
            if drop_missing:
                continue
            raise ValueError(f"Class id {old_id} not in mapping for file {src}")
        parts[0] = str(mapping[old_id])
        out_lines.append(" ".join(parts))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dir", required=True, help="YOLO root directory")
    ap.add_argument("--labels_dir", default="", help="Labels directory (default: yolo_dir/labels)")
    ap.add_argument("--classes_in", default="", help="Existing classes.txt (default: yolo_dir/classes.txt)")
    ap.add_argument("--classes_out", required=True, help="New classes.txt (predefined list)")
    ap.add_argument("--mapping_json", default="", help="Optional JSON mapping old_id -> new_id")
    ap.add_argument("--out_dir", default="", help="Output directory (default: overwrite yolo_dir)")
    ap.add_argument("--drop_missing", action="store_true", help="Drop labels not in mapping")
    ap.add_argument("--write_classes", action="store_true", help="Write new classes.txt to output")
    ap.add_argument("--write_dataset_yaml", action="store_true", help="Write dataset.yaml to output")
    args = ap.parse_args()

    yolo_dir = Path(args.yolo_dir)
    labels_dir = Path(args.labels_dir) if args.labels_dir else yolo_dir / "labels"
    classes_in = Path(args.classes_in) if args.classes_in else yolo_dir / "classes.txt"
    classes_out = Path(args.classes_out)
    out_dir = Path(args.out_dir) if args.out_dir else yolo_dir
    out_labels = out_dir / "labels"

    old_classes = load_classes(classes_in)
    new_classes = load_classes(classes_out)

    mapping = build_name_map(old_classes, new_classes)
    if args.mapping_json:
        mapping.update(load_mapping_json(Path(args.mapping_json)))

    label_files = list(labels_dir.rglob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under: {labels_dir}")

    for src in tqdm(label_files, desc="Labels", unit="file"):
        rel = src.relative_to(labels_dir)
        dst = out_labels / rel
        remap_label_file(src, dst, mapping, args.drop_missing)

    if args.write_classes:
        (out_dir / "classes.txt").write_text("\n".join(new_classes), encoding="utf-8")

    if args.write_dataset_yaml:
        images_root = out_dir / "images"
        has_images = images_root.exists()
        train_path = "images/train" if (images_root / "train").exists() else "images"
        val_path = "images/val" if (images_root / "val").exists() else "images"
        test_path = "images/test" if (images_root / "test").exists() else ""
        if not has_images:
            train_path = "images"
            val_path = "images"
            test_path = ""

        lines = [
            f"path: {out_dir.as_posix()}",
            f"train: {train_path}",
            f"val: {val_path}",
        ]
        if test_path:
            lines.append(f"test: {test_path}")
        lines.append("names:")
        for idx, name in enumerate(new_classes):
            lines.append(f"  {idx}: {name}")

        (out_dir / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
