# ObjectionDetectionAnnotationConverter
Convert Pascal VOC XML object-detection boxes into COCO instance segmentation masks using SAM (Segment Anything).

## What it does
- Reads Pascal VOC XML files with bounding boxes.
- Runs SAM with each box prompt to produce a mask.
- Writes a COCO JSON file with RLE masks, bounding boxes, and areas.

## Requirements
- Python 3.11+
- A SAM checkpoint (.pth) matching the chosen model type

## Install (uv)
```powershell
uv pip install --python .venv\Scripts\python.exe opencv-python numpy pycocotools transformers huggingface-hub tqdm
uv pip install --python .venv\Scripts\python.exe torch torchvision --torch-backend=cu118
```

Or sync from `pyproject.toml`:
```powershell
uv sync
```

## Usage

Convert from object detection annotations to instance segmentation annotations (SAM2 from Hugging Face)
```
python converter.py --xml_dir path\to\xml --img_dir path\to\images --out_json path\to\out\coco.json --sam2_model_id facebook/sam2-hiera-large --device cuda
python converter.py --xml_dir F:\nematoda\AgriNema\Formated_Dataset\VOC2007\Annotations --img_dir F:\nematoda\AgriNema\Formated_Dataset\VOC2007\JPEGImages --out_json F:\nematoda\AgriNema\Formated_Dataset\VOC2007\coco.json --device cuda
```

Update a single XML inside an existing COCO (replace its annotations)
```
python converter.py --xml_dir path\to\xml --xml_file example.xml --img_dir path\to\images --out_json path\to\out\coco.json --sam2_model_id facebook/sam2-hiera-large --device cuda
```

Visualise the converted results (opens a window; press any key for next image, `q`/`Esc` to quit)
```
python visualize_results.py --coco_json path\to\out\coco.json --img_dir path\to\images --panel --show_bbox
```

Convert COCO instance masks to YOLO segmentation labels
```
python coco2yolo.py --coco_json path\to\coco.json --out_dir path\to\yolo --img_dir path\to\images --copy_images
```

Convert COCO to YOLO with train/val/test split
```
python coco2yolo.py --coco_json path\to\coco.json --out_dir path\to\yolo --img_dir path\to\images --copy_images --split --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --seed 42
```

Convert COCO to LabelMe JSON files
```
python coco2labelmejson.py --coco_json path\to\coco.json --out_dir path\to\labelme --img_dir path\to\images --copy_images
```

Remap YOLO classes to a predefined list (and write dataset.yaml)
```
python yolo_remap_classes.py --yolo_dir path\to\yolo --classes_out path\to\new_classes.txt --out_dir path\to\yolo_remap --write_classes --write_dataset_yaml
```

### Options
- `--xml_dir`: directory of Pascal VOC XML files.
- `--img_dir`: directory of corresponding images.
- `--out_json`: output COCO JSON file.
- `--sam2_model_id`: Hugging Face model id (default: `facebook/sam2-hiera-large`).
- `--device`: `cuda` or `cpu` (default: `cuda`).
- `--clip_to_box`: clip masks to the input bbox to reduce spillover.

## Notes
- The first SAM2 run will download weights from Hugging Face into your HF cache.
- If an XML filename does not exist in `--img_dir`, the script tries common extensions.
- Categories are created from XML object names in the order first seen.
- COCO masks are stored as RLE with UTF-8 string counts for JSON.
