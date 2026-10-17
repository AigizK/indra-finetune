"""Explicitly labelled line images (or page images with line ink boxes)."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from indra_ocr.rec_preproc import pad_batch, width_batches
from indra_ocr.vocab import Vocab
from .cache import crop_image, image_hashes, prepare_cache, preprocessing_hash
from .metrics import edit_distance  # backward-compatible import


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
    def __init__(self, rows: list[dict], vocab: Vocab, height: int, stride: int = 4,
                 cache_dir=None):
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
        self._hashes = image_hashes(rows)
        identity = [[self._hashes[r["image"]], r.get("bbox"), vocab.normalize(r["text"])] for r in rows]
        self.signature = hashlib.sha256(json.dumps(
            [preprocessing_hash(height), stride, identity], ensure_ascii=False
        ).encode()).hexdigest()
        self.cached = prepare_cache(rows, height, cache_dir, self._hashes) if cache_dir else None
        self._widths = [entry[1] for entry in self.cached] if self.cached else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, ids = self.rows[index], self.targets[index]
        if self.cached:
            crop = np.load(self.cached[index][0], mmap_mode="r", allow_pickle=False)
        else:
            gray = cv2.imread(row["image"], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"{row['location']}: cannot decode image")
            crop = crop_image(gray, row, self.height)
        steps = (crop.shape[1] + self.stride - 1) // self.stride
        needed = len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))
        if needed > steps:
            raise ValueError(
                f"{row['location']}: CTC needs {needed} steps, image supplies {steps}; "
                "check the transcription/box or split this line"
            )
        return crop, ids, self.vocab.normalize(row["text"])

    @property
    def widths(self):
        if self._widths is None:
            self._widths = [self[i][0].shape[1] for i in range(len(self))]
        return self._widths


class WidthBatchSampler(Sampler):
    """Shuffle ties and batches each epoch, keeping all rows exactly once."""
    def __init__(self, widths, batch_size, generator, shuffle=True, max_ratio=1.3,
                 max_area=40960):
        padded = [(int(w) + 7) // 8 * 8 for w in widths]
        if (batch_size < 1 or not math.isfinite(max_ratio) or max_ratio < 1
                or max_area < max(padded, default=0) or any(w <= 0 for w in widths)):
            raise ValueError("invalid width batching limits (max_area must fit the widest line)")
        self.widths, self.batch_size, self.generator = padded, batch_size, generator
        self.shuffle, self.max_ratio, self.max_area = shuffle, max_ratio, max_area

    def __iter__(self):
        order = (torch.randperm(len(self.widths), generator=self.generator).tolist()
                 if self.shuffle else list(range(len(self.widths))))
        parts = width_batches([self.widths[i] for i in order], self.batch_size,
                              max_ratio=self.max_ratio, max_area=self.max_area)
        batches = [[order[j] for j in part] for part in parts]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=self.generator).tolist()]
        return iter(batches)

    def __len__(self):
        return len(width_batches(self.widths, self.batch_size, max_ratio=self.max_ratio,
                                 max_area=self.max_area))


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
