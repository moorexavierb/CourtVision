"""Convert other basketball datasets to the YOLO layout the training notebooks use.

* COCO json (DeepSportradar instance segmentation, most CVAT / Label Studio exports):
  ``coco_to_yolo``.
* MOT folders (SportsMOT, TeamTrack): see ``sportsmot.py``; notebook 01 converts them
  with pseudo-labels for the classes they lack.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .datasets import write_data_yaml


def coco_to_yolo(annotations: str, images_dir: str, out: str, rename: dict[str, Optional[str]],
                 names: list[str], split: str = "train") -> int:
    """Write a COCO detection set as YOLO under out/<split>/. rename maps COCO category names to
    our class names (None drops). Images with none of our classes are still kept: their empty
    label says "nothing here", which is right only if the set labels every player and ball, so
    drop classes you do not trust rather than leaving them unlabelled. Returns images written."""
    coco = json.load(open(annotations))
    cat = {c["id"]: rename.get(c["name"], c["name"]) for c in coco["categories"]}
    cat = {k: names.index(v) for k, v in cat.items() if v is not None and v in names}
    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        by_image.setdefault(a["image_id"], []).append(a)
    out_p = Path(out)
    (out_p / split / "images").mkdir(parents=True, exist_ok=True)
    (out_p / split / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for im in coco["images"]:
        src = Path(images_dir) / im["file_name"]
        if not src.exists():
            continue
        W, H = im["width"], im["height"]
        stem = Path(im["file_name"]).with_suffix("").as_posix().replace("/", "_")
        rows = []
        for a in by_image.get(im["id"], []):
            if a["category_id"] not in cat:
                continue
            x, y, w, h = a["bbox"]
            rows.append(f"{cat[a['category_id']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        shutil.copy(src, out_p / split / "images" / (stem + src.suffix))
        (out_p / split / "labels" / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n += 1
    write_data_yaml(out_p, names)
    return n
