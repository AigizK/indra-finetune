"""CTC fine-tuning on CPU/CUDA/MPS with baseline metrics and portable checkpoints."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from indra_ocr.vocab import Vocab
from .data import Lines, WidthBatchSampler, collate, decode_batch, read_manifest
from .metrics import TranscriptionMetrics
from .model import load_checkpoint, save_checkpoint


def select_device(requested):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be auto, cpu, cuda or mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def ctc_loss(logits, lengths, targets, target_lengths, criterion):
    logp = logits.float().log_softmax(-1).transpose(0, 1)
    # CopyBackward preserves the gradient to the MPS model. CTC itself runs on
    # CPU because PyTorch has no native MPS CTC kernel.
    if logits.device.type == "mps":
        logp = logp.cpu()
    loss = criterion(logp, targets.to(logp.device), lengths.cpu(), target_lengths.cpu())
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite CTC loss; inspect line boxes and labels")
    return loss


@torch.no_grad()
def evaluate_model(model, loader, vocab, device, normalization="NFC", criterion=None):
    model.eval()
    metrics = TranscriptionMetrics(normalization)
    total_loss = count = 0
    for images, widths, targets, target_lengths, texts in loader:
        logits, lengths = model(images.to(device), widths.to(device))
        predictions = decode_batch(logits, lengths.cpu(), vocab)
        for text, prediction in zip(texts, predictions):
            metrics.add(text, prediction)
        if criterion is not None:
            loss = ctc_loss(logits, lengths, targets, target_lengths, criterion)
            total_loss += loss.item() * len(texts)
            count += len(texts)
    result = metrics.result()
    if criterion is not None:
        result["valid_loss"] = total_loss / count
    return result


def make_loader(dataset, batch_size, generator, shuffle, buckets=True,
                max_ratio=1.3, max_area=40960):
    if buckets:
        sampler = WidthBatchSampler(dataset.widths, batch_size, generator, shuffle,
                                    max_ratio, max_area)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                          num_workers=0, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      num_workers=0, generator=generator)


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_steps < 0:
        raise ValueError("epochs, batch size and learning rate must be positive; max steps >= 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    print(f"device: {device} (CTC: {'cpu' if device == 'mps' else device})", flush=True)
    if device == "mps":
        torch.mps.manual_seed(args.seed)
    model, checkpoint = load_checkpoint(args.checkpoint)
    meta = dict(checkpoint["meta"])
    original_vocab = Vocab.from_json(meta["vocab"])
    vocab = original_vocab
    rows = read_manifest(args.train)
    valid_rows = read_manifest(args.valid) if args.valid else None
    if args.resume and args.extend_vocab:
        raise ValueError("--resume cannot be combined with --extend-vocab")
    if args.extend_vocab:
        extra = sorted(set().union(*(vocab.oov(r["text"]) for r in rows)))
        if extra:
            vocab = Vocab(vocab.chars + extra)
            meta["vocab"] = vocab.to_json()
            print(f"Added {len(extra)} tokens: {extra!r}", flush=True)
    cache_dir = getattr(args, "cache_dir", None)
    train = Lines(rows, vocab, meta["rec_height"], model.config["stride"], cache_dir)
    valid = (Lines(valid_rows, vocab, meta["rec_height"], model.config["stride"], cache_dir)
             if valid_rows else None)
    options = dict(batch_size=args.batch_size, buckets=not getattr(args, "no_width_buckets", False),
                   max_ratio=getattr(args, "max_width_ratio", 1.3),
                   max_area=getattr(args, "max_batch_area", 40960))
    normalization = getattr(args, "normalization", "NFC")
    signatures = {"train": train.signature, "valid": valid.signature if valid else None}
    resume_options = dict(options, normalization=normalization)
    if args.resume:
        if "optimizer" not in checkpoint:
            raise ValueError("--resume needs a training checkpoint with optimizer state")
        if "data_signatures" in checkpoint and checkpoint["data_signatures"] != signatures:
            raise ValueError("training/validation data changed; start without --resume")
        if "loader_options" in checkpoint and checkpoint["loader_options"] != resume_options:
            raise ValueError("batching/normalization settings changed; start without --resume")
        if args.max_steps and checkpoint["step"] >= args.max_steps:
            raise ValueError("checkpoint has already reached --max-steps; increase the limit")
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_loader(train, generator=generator, shuffle=True, **options)
    valid_loader = (make_loader(valid, generator=torch.Generator().manual_seed(args.seed + 1),
                                shuffle=False, **options) if valid else None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "arguments.json").write_text(
        json.dumps(dict(vars(args), resolved_device=device), indent=2), encoding="utf-8")
    model.to(device)
    # Score the input checkpoint BEFORE widening the head or any optimizer step.
    # The loader can carry new token IDs; baseline evaluation uses only its texts.
    start_metrics = (evaluate_model(model, valid_loader, original_vocab, device, normalization)
                     if valid_loader else None)
    baseline = (checkpoint.get("baseline", start_metrics) if args.resume else start_metrics)
    (out / "baseline.json").write_text(json.dumps(
        {"reference": baseline, "at_start": start_metrics,
         "validation_signature": signatures["valid"], "checkpoint": str(args.checkpoint)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if start_metrics:
        print("baseline: " + json.dumps(start_metrics), flush=True)
    if vocab.n_classes != original_vocab.n_classes:
        model.resize_classes(vocab.n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, global_step, best = 0, 0, float("inf")
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["step"]
        # Every run writes its own best.pt, even if the resumed model never
        # improves on an earlier run. Keep the historical metric separately.
        if "rng" in checkpoint:
            torch.set_rng_state(checkpoint["rng"]["torch"])
            generator.set_state(checkpoint["rng"]["loader"])
            if device == "cuda" and checkpoint["rng"].get("cuda"):
                torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
            if device == "mps" and checkpoint["rng"].get("mps") is not None:
                torch.mps.set_rng_state(checkpoint["rng"]["mps"])
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        total_loss, count, stopped = 0.0, 0, False
        started = time.perf_counter()
        for images, widths, targets, target_lengths, texts in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(images.to(device), widths.to(device))
            loss = ctc_loss(logits, lengths, targets, target_lengths, criterion)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True,
                                     foreach=False if device == "mps" else None)
            optimizer.step()
            count += len(texts)
            total_loss += loss.item() * len(texts)
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                stopped = True
                break
        record = {"epoch": epoch + 1, "step": global_step, "train_loss": total_loss / count}
        if valid_loader:
            record.update(evaluate_model(model, valid_loader, vocab, device, normalization, criterion))
            if baseline:
                record["improved_over_baseline"] = record["cer"] < baseline["cer"]
        record["seconds"] = time.perf_counter() - started
        metric = record.get("cer", record["train_loss"])
        improved = metric < best
        best = min(best, metric)
        state = {
            "epoch": epoch + 1, "step": global_step, "best_metric": best,
            "historical_best_metric": min(best, checkpoint.get("historical_best_metric",
                                                               checkpoint.get("best_metric", float("inf")))),
            "optimizer": optimizer.state_dict(), "metrics": record, "baseline": baseline,
            "data_signatures": signatures, "loader_options": resume_options,
            "rng": {"torch": torch.get_rng_state(), "loader": generator.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    "mps": torch.mps.get_rng_state() if device == "mps" else None},
        }
        save_checkpoint(out / "last.pt", model, meta, checkpoint["source"], **state)
        if improved:
            save_checkpoint(out / "best.pt", model, meta, checkpoint["source"], **state)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stopped:
            break
    if not valid_loader:
        print("No validation manifest: best.pt is selected by training loss; CER was not measured.")
