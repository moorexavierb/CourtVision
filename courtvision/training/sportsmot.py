"""Read SportsMOT basketball sequences (MCG-NJU, CC BY-NC 4.0, gated on Hugging Face).

Broadcast basketball clips at 25 fps with every on-court player boxed and tracked, including the
crowded, overlapping and blurred moments where detectors lose players. Layout after download:

    <root>/<split>/<sequence>/img1/000001.jpg
    <root>/<split>/<sequence>/gt/gt.txt        frame,id,x,y,w,h,conf,class,visibility
    <root>/<split>/<sequence>/seqinfo.ini      frameRate=25
    <root>/splits_txt/basketball.txt           which sequences are basketball

The full download is about 23 GB (three sports, and a test split without labels). ``basketball_zip``
keeps only the basketball train and val sequences, in one zip of a few GB that fits in Drive;
``unpack`` extracts it to local disk at the start of a session (much faster to read than Drive).

``to_yolo`` turns the train and val sequences into one YOLO source, one frame in ``stride``, with
files named game__sequence_frame so the dataset build keeps every game on one side of a split
(Source(group_by="prefix")). SportsMOT's own train/val split is not used: its val clips can come
from the same games as its train clips.
"""
from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import cv2
import numpy as np
import yaml


def basketball_sequences(root: str, split: str) -> list[Path]:
    """Sequences of a split (train, val), filtered to basketball when the list file exists."""
    seqs = sorted(p for p in (Path(root) / split).glob("*") if (p / "img1").is_dir())
    listing = Path(root) / "splits_txt" / "basketball.txt"
    if listing.exists():
        keep = {l.strip() for l in open(listing) if l.strip()}
        seqs = [s for s in seqs if s.name in keep]
    return seqs


def game_of(seq: Path | str) -> str:
    """The broadcast a sequence was cut from: v_<youtube id>_c<clip> -> v_<youtube id>. Clips of one
    game share arena, teams and camera, so they are kept on one side of every split."""
    return re.sub(r"_c\d+$", "", Path(seq).name)


def read_tracks(seq: Path) -> dict[int, dict[int, list[float]]]:
    """{frame: {track id: [x0, y0, x1, y1]}} from gt/gt.txt (rows marked ignore are skipped)."""
    out: dict[int, dict[int, list[float]]] = {}
    gt = Path(seq) / "gt" / "gt.txt"
    if not gt.exists():
        return out
    for line in open(gt):
        p = line.strip().split(",")
        if len(p) < 6 or (len(p) > 6 and float(p[6]) == 0):
            continue
        f, tid = int(float(p[0])), int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        out.setdefault(f, {})[tid] = [x, y, x + w, y + h]
    return out


def player_boxes(seq: Path) -> dict[int, np.ndarray]:
    """{frame: (N, 4) player boxes}."""
    return {f: np.array(list(b.values()), float) for f, b in read_tracks(seq).items()}


def frame_paths(seq: Path) -> dict[int, Path]:
    return {int(p.stem): p for p in (Path(seq) / "img1").glob("*.jpg")}


def seq_fps(seq: Path, default: float = 25.0) -> float:
    ini = Path(seq) / "seqinfo.ini"
    if ini.exists():
        for line in open(ini):
            if line.lower().startswith("framerate"):
                return float(line.split("=")[1])
    return default


def to_yolo(root: str, out: str, class_names: list[str], splits=("train", "val"), stride: int = 10,
            fill: Optional[Callable[[np.ndarray, np.ndarray], Optional[tuple]]] = None) -> dict:
    """Write every stride-th frame of the basketball sequences as a YOLO dataset in out/train.

    fill(image, player_xyxy) -> (xyxy, class ids) adds the classes SportsMOT does not label (see
    labels.fill_missing), or returns None to skip an ambiguous frame. Without fill, frames get
    player boxes only (class 0 = class_names[0], which must be the player class).
    Returns {"frames": written, "skipped": skipped by fill, "sequences": n, "games": n}."""
    img_dir, lbl_dir = Path(out) / "train" / "images", Path(out) / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    from .labels import box_rows
    counts = {"frames": 0, "skipped": 0, "sequences": 0, "games": set()}
    for split in splits:
        for seq in basketball_sequences(root, split):
            boxes, paths = player_boxes(seq), frame_paths(seq)
            counts["sequences"] += 1
            counts["games"].add(game_of(seq))
            for f in sorted(boxes)[::stride]:
                img = cv2.imread(str(paths[f])) if f in paths else None
                if img is None:
                    continue
                labels = fill(img, boxes[f]) if fill else (boxes[f], np.zeros(len(boxes[f]), int))
                if labels is None:
                    counts["skipped"] += 1
                    continue
                stem = f"{game_of(seq)}__{seq.name}_{f:06d}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
                rows = box_rows(*labels, (img.shape[1], img.shape[0]))
                (lbl_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                counts["frames"] += 1
    yaml.safe_dump({"names": {i: n for i, n in enumerate(class_names)}}, open(Path(out) / "data.yaml", "w"))
    counts["games"] = len(counts["games"])
    return counts


def sequences_of_games(root: str, games: set[str], splits=("train", "val")) -> list[Path]:
    """Full sequences (every frame) of the given games, e.g. the games a build put in test."""
    return [seq for split in splits for seq in basketball_sequences(root, split) if game_of(seq) in games]


# ----------------------------------------------------------------- download -> basketball-only zip

def _archive_files(path: Path):
    """(name, readable file) for every file in a zip or tar (.tar, .tar.gz), in archive order."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir():
                    with z.open(info) as f:
                        yield info.filename, f
    else:
        with tarfile.open(path, "r|*") as t:  # streaming read: one pass, also for .tar.gz
            for m in t:
                if m.isfile():
                    yield m.name, t.extractfile(m)


def _dir_files(root: Path):
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            with open(p, "rb") as f:
                yield p.relative_to(root).as_posix(), f


def _sources(raw_dir: Path) -> list:
    """The downloaded archives, or the folder itself when it holds extracted files."""
    archives = [p for p in sorted(Path(raw_dir).rglob("*")) if p.is_file() and ".cache" not in p.parts
                and (zipfile.is_zipfile(p) or p.name.endswith((".tar", ".tar.gz", ".tgz")))]
    return [lambda p=p: _archive_files(p) for p in archives] or [lambda: _dir_files(Path(raw_dir))]


def basketball_zip(raw_dir: str, out_zip: str, splits=("train", "val")) -> dict:
    """Copy the basketball sequences of the given splits, plus splits_txt/, from a SportsMOT
    download (zips, tars or extracted folders) into one zip laid out as
    train/<seq>/..., val/<seq>/..., splits_txt/basketball.txt. Images are stored, not recompressed."""
    sources = _sources(Path(raw_dir))
    keep = None
    for files in sources:  # pass 1: the list of basketball sequences
        for name, f in files():
            if name.endswith("splits_txt/basketball.txt"):
                keep = {l.strip() for l in f.read().decode().splitlines() if l.strip()}
                break
        if keep:
            break
    if not keep:
        raise FileNotFoundError("splits_txt/basketball.txt is not in the download")
    seqs, n = set(), 0
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as out:
        for files in sources:  # pass 2: copy what is kept
            for name, f in files():
                parts = PurePosixPath(name).parts
                arc = None
                if "splits_txt" in parts:
                    arc = "splits_txt/" + parts[-1]
                else:
                    for i, part in enumerate(parts[:-1]):
                        if part in splits and parts[i + 1] in keep:
                            arc = "/".join(parts[i:])
                            seqs.add((part, parts[i + 1]))
                            break
                if arc:
                    out.writestr(arc, f.read())
                    n += 1
    return {"files": n, "sequences": {s: sum(1 for sp, _ in seqs if sp == s) for s in splits},
            "basketball_listed": len(keep), "zip_gb": round(Path(out_zip).stat().st_size / 1e9, 2)}


def unpack(zip_path: str, dest: str) -> str:
    """Extract the basketball zip to dest (skipped when already there); returns dest."""
    if not (Path(dest) / "splits_txt" / "basketball.txt").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return dest
