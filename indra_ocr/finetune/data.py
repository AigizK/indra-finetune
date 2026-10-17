"""Explicitly labelled line images (or page images with line ink boxes)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from indra_ocr.rec_preproc import crop_line, pad_batch
from indra_ocr.vocab import Vocab


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row["text"], str) or not isinstance(row["image"], str):
                raise ValueError("image and text must be strings")
            row["image"] = str((path.parent / row["image"]).resolve())
            if not Path(row["image"]).is_file():
                raise ValueError(f"image does not exist: {row['image']}")
            if "bbox" in row:
                box = np.asarray(row["bbox"], dtype=float)
                if (box.shape != (4,) or not np.isfinite(box).all()
                        or box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError("bbox must be finite [x0,y0,x1,y1] with positive size")
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        row["location"] = f"{path}:{number}"
        rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


class Lines(Dataset):
    def __init__(self, rows: list[dict], vocab: Vocab, height: int, stride: int = 4):
        self.rows, self.vocab, self.height, self.stride = rows, vocab, height, stride
        self.targets = []
        for row in rows:
            ids = vocab.encode(row["text"])
            if ids is None:
                raise ValueError(
                    f"{row['location']}: empty text or unknown tokens: "
                    f"{sorted(vocab.oov(row['text']))!r}; use --extend-vocab for training"
                )
            self.targets.append(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, ids = self.rows[index], self.targets[index]
        gray = cv2.imread(row["image"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"{row['location']}: cannot decode image")
        h, w = gray.shape
        box = row.get("bbox", [0, 0, w, h])
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
            raise ValueError(f"{row['location']}: bbox is outside the image")
        crop = crop_line(gray, box, height=self.height)
        steps = (crop.shape[1] + self.stride - 1) // self.stride
        needed = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if needed > steps:
            raise ValueError(
                f"{row['location']}: CTC needs {needed} steps, image supplies {steps}; "
                "check the transcription/box or split this line"
            )
        return crop, ids, self.vocab.normalize(row["text"])


def collate(items):
    crops, labels, texts = zip(*items)
    images, widths = pad_batch(list(crops))
    return (
        torch.from_numpy(images), torch.from_numpy(widths),
        torch.tensor([i for ids in labels for i in ids], dtype=torch.long),
        torch.tensor([len(ids) for ids in labels], dtype=torch.long), list(texts),
    )


def decode_batch(logits: torch.Tensor, lengths: torch.Tensor, vocab: Vocab) -> list[str]:
    result = []
    for seq, length in zip(logits.detach().argmax(-1).cpu().tolist(), lengths.tolist()):
        ids, previous = [], -1
        for token in seq[:length]:
            if token != 0 and token != previous:
                ids.append(token)
            previous = token
        result.append(vocab.decode(ids))
    return result


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]
