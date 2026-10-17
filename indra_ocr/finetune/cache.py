"""Content-addressed, lossless line cache; only one source image is decoded at a time."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from indra_ocr import rec_preproc


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def crop_image(gray, row, height):
    h, w = gray.shape
    box = row.get("bbox", [0, 0, w, h])
    if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
        raise ValueError(f"{row['location']}: bbox is outside the image")
    return rec_preproc.crop_line(gray, box, height=height)


def image_hashes(rows):
    return {path: file_hash(path) for path in dict.fromkeys(r["image"] for r in rows)}


def preprocessing_hash(height):
    return file_hash(rec_preproc.__file__) + f":height={height}"


def prepare_cache(rows, height, directory, hashes):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rule = preprocessing_hash(height)
    entries, by_image = [None] * len(rows), {}
    for i, row in enumerate(rows):
        by_image.setdefault(row["image"], []).append(i)
    for image, indices in by_image.items():
        gray = None
        for i in indices:
            row = rows[i]
            key = hashlib.sha256(json.dumps([hashes[image], row.get("bbox"), rule],
                                             sort_keys=True).encode()).hexdigest()
            path = directory / f"{key}.npy"
            if not path.exists():
                if gray is None:
                    gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        raise ValueError(f"{row['location']}: cannot decode image")
                crop = crop_image(gray, row, height)
                with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    try:
                        np.save(stream, crop, allow_pickle=False)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                temporary.replace(path)
            # mmap reads the header without loading the full dataset into RAM.
            crop = np.load(path, mmap_mode="r", allow_pickle=False)
            if crop.dtype != np.float32 or crop.ndim != 2 or crop.shape[0] != height:
                raise ValueError(f"invalid line cache entry: {path}; remove it and retry")
            entries[i] = (path, crop.shape[1])
    return entries
