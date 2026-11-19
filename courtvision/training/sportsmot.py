"""Read SportsMOT basketball sequences (MCG-NJU, CC BY-NC 4.0, gated on Hugging Face).

Broadcast basketball clips at 25 fps with every on-court player boxed and tracked, including the
crowded, overlapping and blurred moments where detectors lose players. Layout after download:

    <root>/<split>/<sequence>/img1/000001.jpg
    <root>/<split>/<sequence>/gt/gt.txt        frame,id,x,y,w,h,conf,class,visibility
    <root>/<split>/<sequence>/seqinfo.ini      frameRate=25
    <root>/splits_txt/basketball.txt           which sequences are basketball

The full download is about 23 GB (three sports, and a test split without labels). ``basketball_zip``
keeps only the basketball train and val sequences, in one zip of a few GB that fits in Drive,
reading the train and val archives once each and never the test one; ``unpack`` extracts it to
local disk at the start of a session (much faster to read than Drive).

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


# splits_txt/basketball.txt of the SportsMOT release (train, val and test sequences): used when the
# list file is not next to the data, e.g. a Drive copy where splits_txt/ sits outside the tars.
BASKETBALL = frozenset("""
    v_-6Os86HzwCs_c001 v_-6Os86HzwCs_c003 v_-6Os86HzwCs_c007 v_-6Os86HzwCs_c009
    v_-9kabh1K8UA_c008 v_-9kabh1K8UA_c009 v_-9kabh1K8UA_c010 v_00HRwkvvjtQ_c001
    v_00HRwkvvjtQ_c003 v_00HRwkvvjtQ_c005 v_00HRwkvvjtQ_c007 v_00HRwkvvjtQ_c008
    v_00HRwkvvjtQ_c011 v_2BhBRkkAqbQ_c002 v_2j7kLB-vEEk_c001 v_2j7kLB-vEEk_c002
    v_2j7kLB-vEEk_c005 v_2j7kLB-vEEk_c007 v_2j7kLB-vEEk_c009 v_2j7kLB-vEEk_c010
    v_42VrMbd68Zg_c003 v_42VrMbd68Zg_c004 v_42VrMbd68Zg_c011 v_42VrMbd68Zg_c012
    v_4LXTUim5anY_c002 v_4LXTUim5anY_c003 v_4LXTUim5anY_c010 v_4LXTUim5anY_c012
    v_4LXTUim5anY_c013 v_4r8QL_wglzQ_c001 v_5ekaksddqrc_c001 v_5ekaksddqrc_c002
    v_5ekaksddqrc_c003 v_5ekaksddqrc_c004 v_5ekaksddqrc_c005 v_6OLC1-bhioc_c001
    v_6OLC1-bhioc_c003 v_6OLC1-bhioc_c005 v_6OLC1-bhioc_c007 v_6OLC1-bhioc_c008
    v_6OLC1-bhioc_c009 v_6OLC1-bhioc_c011 v_6OLC1-bhioc_c012 v_6OLC1-bhioc_c013
    v_6OLC1-bhioc_c014 v_6oTxfzKdG6Q_c004 v_6oTxfzKdG6Q_c015 v_7FTsO8S3h88_c001
    v_7FTsO8S3h88_c004 v_7FTsO8S3h88_c005 v_7FTsO8S3h88_c006 v_7FTsO8S3h88_c007
    v_7FTsO8S3h88_c008 v_82WN8C8qJmI_c002 v_82WN8C8qJmI_c003 v_82WN8C8qJmI_c004
    v_82WN8C8qJmI_c005 v_82WN8C8qJmI_c014 v_A4OJhlI6hgc_c001 v_A4OJhlI6hgc_c002
    v_A4OJhlI6hgc_c003 v_A4OJhlI6hgc_c005 v_A4OJhlI6hgc_c601 v_BdD9xu0E2H4_c003
    v_BdD9xu0E2H4_c004 v_BdD9xu0E2H4_c006 v_BdD9xu0E2H4_c010 v_BdD9xu0E2H4_c011
    v_BdD9xu0E2H4_c013 v_BgwzTUxJaeU_c008 v_BgwzTUxJaeU_c012 v_BgwzTUxJaeU_c014
    v_aVfJwdQxCsU_c001 v_aVfJwdQxCsU_c007 v_aVfJwdQxCsU_c015 v_bQNDRprvpus_c001
    v_bQNDRprvpus_c003 v_bQNDRprvpus_c005 v_bQNDRprvpus_c007 v_bQNDRprvpus_c008
""".split())


def basketball_list(root: str | Path) -> set[str]:
    """Basketball sequence names: <root>/splits_txt/basketball.txt (or one level up) if present,
    else the copy built into this module."""
    for d in (Path(root), Path(root).parent):
        listing = d / "splits_txt" / "basketball.txt"
        if listing.exists():
            return {l.strip() for l in open(listing) if l.strip()}
    return set(BASKETBALL)


def basketball_sequences(root: str, split: str) -> list[Path]:
    """Basketball sequences of a split (train, val)."""
    keep = basketball_list(root)
    return sorted(p for p in (Path(root) / split).glob("*") if (p / "img1").is_dir() and p.name in keep)


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

ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".zip")
ALL_SPLITS = ("train", "val", "test")


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
        if p.is_file() and not p.name.endswith(ARCHIVE_EXTS):
            with open(p, "rb") as f:
                yield p.relative_to(root).as_posix(), f


def _split_of_archive(path: Path) -> Optional[str]:
    """train.tar -> "train"; None for an archive not named after one split."""
    stem = path.name
    for ext in ARCHIVE_EXTS:
        stem = stem[: -len(ext)] if stem.endswith(ext) else stem
    return stem if stem in ALL_SPLITS else None


def archives(raw_dir: str | Path, splits=("train", "val")) -> list[Path]:
    """The archives of a download that can hold the given splits. An archive named after another
    split (test.tar: 23 GB, no labels) is left out, so it is never read."""
    found = [p for p in sorted(Path(raw_dir).rglob("*")) if p.is_file() and ".cache" not in p.parts
             and p.name.endswith(ARCHIVE_EXTS) and not p.name.endswith(".incomplete")]
    return [p for p in found if _split_of_archive(p) in (None, *splits)]


def _target(name: str, keep: set[str], splits, archive_split: Optional[str]) -> Optional[str]:
    """Where a file of the download goes in the basketball zip (split/sequence/...), or None.
    The split comes from the path (.../train/<seq>/...) or, when the path has none, from the
    archive's name (train.tar holding <seq>/img1/...)."""
    parts = [x for x in PurePosixPath(name).parts if x not in (".", "")]
    for i, part in enumerate(parts[:-1]):
        if part in keep:
            split = parts[i - 1] if i and parts[i - 1] in ALL_SPLITS else archive_split
            if split in splits:
                return "/".join([split, *parts[i:]])
            return None
    return None


def find_basketball_list(raw_dir: str | Path) -> set[str]:
    """splits_txt/basketball.txt as a loose file anywhere in raw_dir, else inside a zip there
    (a zip's file list is cheap to read; a tar would need a full pass), else the built-in copy."""
    raw = Path(raw_dir)
    for listing in sorted(raw.rglob("basketball.txt")):
        if listing.parent.name == "splits_txt":
            return {l.strip() for l in open(listing) if l.strip()}
    for a in archives(raw, ALL_SPLITS):
        if zipfile.is_zipfile(a):
            with zipfile.ZipFile(a) as z:
                for name in z.namelist():
                    if name.endswith("splits_txt/basketball.txt"):
                        return {l.strip() for l in z.read(name).decode().splitlines() if l.strip()}
    return set(BASKETBALL)


def basketball_zip(raw_dir: str, out_zip: str, splits=("train", "val")) -> dict:
    """Copy the basketball sequences of the given splits from a SportsMOT download (tars, zips or
    extracted folders, e.g. Drive's sportsmot/dataset/{train,val}.tar) into one zip laid out as
    train/<seq>/..., val/<seq>/..., splits_txt/basketball.txt. One streaming pass over each
    archive; images are stored, not recompressed.

    The basketball list comes from find_basketball_list: a loose splits_txt/ folder next to the
    tars is fine, and without one the copy in this module is used."""
    raw = Path(raw_dir)
    keep = find_basketball_list(raw)
    seqs, n = set(), 0
    Path(out_zip).parent.mkdir(parents=True, exist_ok=True)
    sources = [(lambda a=a: _archive_files(a), _split_of_archive(a)) for a in archives(raw, splits)]
    sources.append((lambda: _dir_files(raw), None))  # sequences already extracted, if any
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as out:
        out.writestr("splits_txt/basketball.txt", "\n".join(sorted(keep)) + "\n")
        written = {"splits_txt/basketball.txt"}
        for files, archive_split in sources:
            for name, f in files():
                arc = _target(name, keep, splits, archive_split)
                if arc and arc not in written:
                    out.writestr(arc, f.read())
                    written.add(arc)
                    seqs.add(tuple(arc.split("/")[:2]))
                    n += 1
    return {"files": n, "sequences": {s: sum(1 for sp, _ in seqs if sp == s) for s in splits},
            "basketball_listed": len(keep), "zip_gb": round(Path(out_zip).stat().st_size / 1e9, 2)}


def check_zip(zip_path: str, expected: dict) -> bool:
    """A zip is complete when it opens and holds the sequences basketball_zip reported."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    seqs = {tuple(n.split("/")[:2]) for n in names if "/img1/" in n}
    return all(sum(1 for sp, _ in seqs if sp == s) == k for s, k in expected.items())


def unpack(zip_path: str, dest: str) -> str:
    """Extract the basketball zip to dest (skipped when already there); returns dest."""
    if not (Path(dest) / "splits_txt" / "basketball.txt").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return dest
