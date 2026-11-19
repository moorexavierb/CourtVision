"""Build one training set from several datasets, with no leakage between splits.

Every image keeps its source dataset, and images that come from the same footage are tied
together so they always land in the same split. Two images are tied when

1. they come from the same video: Roboflow keeps the original file name in front of
   ``_jpg.rf.<hash>``, so augmented copies of one frame and frames of one clip share a key
   (``video_key``); SportsMOT frames are named after their sequence; or
2. they look the same (difference hash within ``near_dup_bits`` bits), whatever their names or
   datasets: a frame re-exported by a fork of a dataset, the same broadcast moment in two
   datasets, near-identical neighbouring frames.

Tied images form a *component*. Splits are drawn by component, separately for each source, so
no footage is on both sides. Sources marked ``role="test"`` (clips kept back to score the models)
pull every component they touch into the test split, so nothing near a test image is trained on.

Layout written to ``out``:

    images/<source>/...  labels/<source>/...   YOLO files, our class ids
    manifest.csv                                 one row per image: source, component, split, boxes
    lists/{train,val,test}.txt, data.yaml        everything
    data_<source>.yaml                           validate / test on one source
    folds/lodo_<source>.yaml                     leave-one-dataset-out (lodo_folds)
    folds/kfold_<i>.yaml                         grouped k-fold (kfold_folds)

Works for detection (labels "cls cx cy w h") and court keypoints (pose rows, kpt_shape [18, 3]).
"""
from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DETECT_CLASSES = ["Player", "Ref", "Ball", "Hoop"]
# Class names other datasets use for ours (lower case). Anything else is dropped.
SYNONYMS = {"player": "Player", "players": "Player", "ref": "Ref", "referee": "Ref", "referees": "Ref",
            "ball": "Ball", "basketball": "Ball", "sports ball": "Ball", "hoop": "Hoop", "basket": "Hoop",
            "rim": "Hoop", "court": "court"}

_RF_SUFFIX = re.compile(r"(_(jpg|jpeg|png|webp|bmp))?\.rf\.[0-9a-f]{16,}$", re.I)
_FRAME_SUFFIX = re.compile(r"[_\-. ]?(frame|frm|img|image|f)?[_\-. ]?\d+$", re.I)


@dataclass
class Source:
    name: str                       # short id: letters, digits, - or _
    root: str                       # dataset root (Roboflow YOLO export), or a folder of images for negatives
    role: str = "train"             # train | test (never trained on) | negatives (no objects; train only)
    weight: int = 1                 # repeat its training images this many times
    rename: Optional[dict[str, Optional[str]]] = None  # extra class renames (None drops); SYNONYMS apply first
    group_by: str = "auto"          # auto | video | image | prefix  (see assign_groups)


# --------------------------------------------------------------------------- files

def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_for(image: Path) -> Path:
    """Roboflow layout: <split>/images/x.jpg -> <split>/labels/x.txt."""
    return image.parent.parent / "labels" / (image.stem + ".txt")


def class_names(data_yaml: Path) -> list[str]:
    names = yaml.safe_load(open(data_yaml))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def source_images(src: Source) -> list[tuple[Path, Optional[Path]]]:
    """(image, label file or None) for every image of a source, over all its splits."""
    root = Path(src.root)
    if src.role == "negatives":
        return [(p, None) for p in images_in(root)]
    return [(p, label_for(p)) for s in SPLITS if (root / s / "images").exists() for p in images_in(root / s / "images")]


def class_map(src_names: list[str], names: list[str], rename: Optional[dict] = None) -> dict[int, int]:
    """{source class id: our class id}. rename first, then SYNONYMS, then exact names."""
    out = {}
    for i, n in enumerate(src_names):
        target = (rename or {}).get(n, SYNONYMS.get(n.lower(), n))
        if target is not None and target in names:
            out[i] = names.index(target)
    return out


# --------------------------------------------------------------------------- grouping

def original_stem(stem: str) -> str:
    """File stem without Roboflow's export suffix, so augmented copies share one name."""
    return _RF_SUFFIX.sub("", stem)


def video_key(stem: str) -> str:
    """The clip a frame came from: the original stem without a trailing frame number."""
    base = original_stem(stem)
    return _FRAME_SUFFIX.sub("", base) or base


def assign_groups(stems: list[str], group_by: str = "auto", min_groups: int = 10) -> list[str]:
    """Group per image. auto: by video when that gives at least min_groups groups, else by
    original image (augmented copies still stay together). prefix: everything before "__" in the
    file name, for converters that know the game a frame came from (SportsMOT: game__clip_frame)."""
    if group_by == "prefix":
        return [s.split("__")[0] for s in stems]
    if group_by == "image":
        return [original_stem(s) for s in stems]
    videos = [video_key(s) for s in stems]
    if group_by == "video" or len(set(videos)) >= min_groups:
        return videos
    return [original_stem(s) for s in stems]


def hold_out_groups(stems: list[str], frac: float, seed: int = 0, group_by: str = "auto") -> np.ndarray:
    """Mask of images to keep back, whole groups (videos) at a time, about frac of the images.
    Used before pseudo-labelling a dataset, so the frames scored later were never labelled by
    the model being replaced. Pass the kept-back images as a role="test" Source to build, which
    also pulls any near-duplicates of them into test."""
    groups = np.array(assign_groups(stems, group_by))
    order = list(dict.fromkeys(groups))
    random.Random(seed).shuffle(order)
    held, n = set(), 0
    for g in order:
        if n >= frac * len(stems):
            break
        held.add(g)
        n += int((groups == g).sum())
    return np.isin(groups, list(held))


def dhash(path, size: int = 8) -> int:
    """64-bit difference hash. Survives re-encoding and resizing."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def near_duplicate_pairs(hashes: np.ndarray, max_bits: int = 6, chunk: int = 256) -> list[tuple[int, int]]:
    """Index pairs (i < j) whose hashes differ in at most max_bits bits."""
    b = np.asarray(hashes, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    pairs = []
    for s in range(0, len(b), chunk):
        dist = _POPCOUNT[np.bitwise_xor(b[s:s + chunk, None, :], b[None, :, :])].sum(axis=2, dtype=np.int32)
        for i, j in zip(*np.nonzero(dist <= max_bits)):
            if s + i < j:
                pairs.append((int(s + i), int(j)))
    return pairs


def components(groups: list[str], pairs: list[tuple[int, int]]) -> list[int]:
    """Connected components over 'same group' and 'near duplicate' ties (union-find)."""
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for i, g in enumerate(groups):
        union(first.setdefault(g, i), i)
    for a, b in pairs:
        union(a, b)
    roots = [find(i) for i in range(len(groups))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def split_components(manifest: pd.DataFrame, val_frac: float, test_frac: float, seed: int = 0) -> pd.Series:
    """Split per image. A component touching a test source goes to test. Otherwise each source's
    components (owned by the source of their first image) are shuffled and dealt to test, then
    valid, until each holds its share of that source's images; the rest train. A component bigger
    than twice a split's share trains, so one big video cannot become most of a test split."""
    split = {}
    test_comps = set(manifest[manifest.role == "test"].component)
    for c in test_comps:
        split[c] = "test"
    owner = manifest.groupby("component").source.first()
    size = manifest.groupby("component").size()
    rng = random.Random(seed)
    for src in manifest.source.unique():
        role = manifest[manifest.source == src].role.iloc[0]
        comps = sorted(c for c in owner[owner == src].index if c not in split)
        if role == "negatives":
            split.update({c: "train" for c in comps})
            continue
        rng.shuffle(comps)
        total = sum(size[c] for c in comps)
        quotas = [("test", test_frac * total), ("valid", val_frac * total)]
        k, filled = 0, 0
        for i, c in enumerate(comps):
            while k < len(quotas) and filled >= quotas[k][1]:
                k, filled = k + 1, 0
            fits = k < len(quotas) and size[c] <= 2 * quotas[k][1]
            if fits and i < len(comps) - 1:  # the last component always trains
                split[c], filled = quotas[k][0], filled + size[c]
            else:
                split[c] = "train"
    return manifest.component.map(split)


# --------------------------------------------------------------------------- build

def build(sources: list[Source], names: list[str], out: str, val_frac: float = 0.15, test_frac: float = 0.15,
          seed: int = 0, near_dup_bits: int = 6, kpt_shape: Optional[list[int]] = None,
          flip_idx: Optional[list[int]] = None) -> pd.DataFrame:
    """Copy every source into `out` with our class ids, tie and split, write lists and yamls.
    Returns the manifest. For court keypoints pass kpt_shape=[18, 3] and flip_idx."""
    out_root = Path(out)
    shutil.rmtree(out_root, ignore_errors=True)
    rows = []
    for src in sources:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", src.name):
            raise ValueError(f"source name '{src.name}': use letters, digits, - or _")
        items = source_images(src)
        cmap = {} if src.role == "negatives" else class_map(class_names(Path(src.root) / "data.yaml"), names, src.rename)
        groups = assign_groups([img.stem for img, _ in items], src.group_by)
        (out_root / "images" / src.name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / src.name).mkdir(parents=True, exist_ok=True)
        used = set()
        for (img, lbl), group in zip(items, groups):
            stem = img.stem
            while stem in used:  # same file name in two of the source's splits
                stem += "_"
            used.add(stem)
            counts = {n: 0 for n in names}
            lines = []
            if lbl is not None and lbl.exists():
                for line in open(lbl):
                    parts = line.split()
                    if parts and int(parts[0]) in cmap:
                        cls = cmap[int(parts[0])]
                        counts[names[cls]] += 1
                        lines.append(" ".join([str(cls)] + parts[1:]))
            dst = out_root / "images" / src.name / (stem + img.suffix.lower())
            shutil.copy(img, dst)
            (out_root / "labels" / src.name / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append({"image": str(dst), "source": src.name, "role": src.role, "weight": src.weight,
                         "group": f"{src.name}/{group}", **counts})
    m = pd.DataFrame(rows)
    pairs = near_duplicate_pairs(np.array([dhash(p) for p in m.image], dtype=np.uint64), near_dup_bits) if near_dup_bits >= 0 else []
    m["component"] = components(m.group.tolist(), pairs)
    m["split"] = split_components(m, val_frac, test_frac, seed)
    m.attrs["near_duplicate_pairs"] = len(pairs)
    m.to_csv(out_root / "manifest.csv", index=False)
    write_lists(m, names, out_root, kpt_shape, flip_idx)
    return m


def _write_list(path: Path, images) -> str:
    images = list(images)
    path.write_text("\n".join(images) + ("\n" if images else ""))
    return str(path)


def _weighted(df: pd.DataFrame) -> list[str]:
    return [p for p, w in zip(df.image, df.weight) for _ in range(int(w))]


def write_yaml(path: Path, root: Path, names: list[str], train: str, val: str, test: Optional[str] = None,
               kpt_shape=None, flip_idx=None) -> Path:
    cfg = {"path": str(root), "train": train, "val": val}
    if test:
        cfg["test"] = test
    cfg["names"] = {i: n for i, n in enumerate(names)}
    if kpt_shape:
        cfg["kpt_shape"] = list(kpt_shape)
    if flip_idx:
        cfg["flip_idx"] = list(flip_idx)
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    return path


def write_lists(m: pd.DataFrame, names: list[str], root: Path, kpt_shape=None, flip_idx=None) -> None:
    """Training lists and yamls from a manifest (also call after adding rows, e.g. zoomed crops)."""
    lists = root / "lists"
    lists.mkdir(exist_ok=True)
    kw = {"kpt_shape": kpt_shape, "flip_idx": flip_idx}
    tr = _write_list(lists / "train.txt", _weighted(m[m.split == "train"]))
    va = _write_list(lists / "val.txt", m[m.split == "valid"].image)
    te = _write_list(lists / "test.txt", m[m.split == "test"].image)
    write_yaml(root / "data.yaml", root, names, tr, va, te, **kw)
    for name, df in m.groupby("source"):
        v = _write_list(lists / f"val_{name}.txt", df[df.split == "valid"].image)
        t = _write_list(lists / f"test_{name}.txt", df[df.split == "test"].image)
        write_yaml(root / f"data_{name}.yaml", root, names, tr, v if (df.split == "valid").any() else t, t, **kw)
    (root / "build.yaml").write_text(yaml.safe_dump({"names": names, "kpt_shape": kpt_shape, "flip_idx": flip_idx}))


def leakage_report(m: pd.DataFrame) -> dict:
    """The first three should be zero: components or groups split across splits, images of test
    sources outside test. The rest are for reading: how many images were tied to an image of
    another dataset, and the largest component's share of its source (a big share means many
    frames look alike, and that source's splits are coarse)."""
    comp_sources = m.groupby("component").source.nunique()
    biggest = m.groupby(["source", "component"]).size().groupby("source").max() / m.groupby("source").size()
    return {"components_in_two_splits": int((m.groupby("component").split.nunique() > 1).sum()),
            "groups_in_two_splits": int((m.groupby("group").split.nunique() > 1).sum()),
            "test_source_images_not_in_test": int(((m.role == "test") & (m.split != "test")).sum()),
            "images_tied_across_datasets": int(m.component.isin(comp_sources[comp_sources > 1].index).sum()),
            "largest_component_share": {k: round(float(v), 3) for k, v in biggest.items()}}


def summary(m: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Images and labels per class, for each source and split."""
    agg = {"image": "count", **{n: "sum" for n in names if n in m}}
    return m.groupby(["source", "split"]).agg(agg).rename(columns={"image": "images"})


# --------------------------------------------------------------------------- cross-validation

def lodo_folds(root: str, min_images: int = 50) -> dict[str, Path]:
    """Leave-one-dataset-out. For each training source S: train (and early-stop on valid) using
    the other sources, test on every image of S. Components that contain any image of S are left
    out of training, so near-copies of S in other datasets cannot leak in. Returns {S: yaml}."""
    root_p = Path(root)
    m = pd.read_csv(root_p / "manifest.csv")
    meta = yaml.safe_load(open(root_p / "build.yaml"))
    folds = root_p / "folds"
    folds.mkdir(exist_ok=True)
    out = {}
    for held in sorted(m[m.role == "train"].source.unique()):
        mine = m[m.source == held]
        touched = set(mine.component)
        rest = m[(m.source != held) & (m.role != "test") & ~m.component.isin(touched)]
        if len(mine) < min_images or rest[rest.split == "train"].empty:
            continue
        tr = _write_list(folds / f"lodo_{held}_train.txt", _weighted(rest[rest.split == "train"]))
        va = _write_list(folds / f"lodo_{held}_val.txt", rest[rest.split == "valid"].image)
        te = _write_list(folds / f"lodo_{held}_test.txt", mine.image)
        out[held] = write_yaml(folds / f"lodo_{held}.yaml", root_p, meta["names"], tr, va, te,
                               meta.get("kpt_shape"), meta.get("flip_idx"))
    return out


def kfold_folds(root: str, k: int = 3, seed: int = 0) -> list[Path]:
    """Grouped k-fold over the pooled train + valid images: whole components per fold, folds of
    similar size. Fold i validates on part i and trains on the rest; test is never touched."""
    root_p = Path(root)
    m = pd.read_csv(root_p / "manifest.csv")
    meta = yaml.safe_load(open(root_p / "build.yaml"))
    pool = m[m.split.isin(["train", "valid"])]
    sizes = pool.component.value_counts()
    comps = list(sizes.index)
    random.Random(seed).shuffle(comps)
    fold_of, load = {}, [0] * k
    for c in sorted(comps, key=lambda c: -sizes[c]):
        i = load.index(min(load))
        fold_of[c], load[i] = i, load[i] + int(sizes[c])
    folds = root_p / "folds"
    folds.mkdir(exist_ok=True)
    out = []
    for i in range(k):
        part = pool.component.map(fold_of) == i
        tr = _write_list(folds / f"kfold_{i}_train.txt", _weighted(pool[~part]))
        va = _write_list(folds / f"kfold_{i}_val.txt", pool[part].image)
        out.append(write_yaml(folds / f"kfold_{i}.yaml", root_p, meta["names"], tr, va, None,
                              meta.get("kpt_shape"), meta.get("flip_idx")))
    return out
