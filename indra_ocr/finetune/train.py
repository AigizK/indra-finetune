"""CTC fine-tuning with validation CER and portable CPU checkpoints."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from indra_ocr.vocab import Vocab
from .data import Lines, collate, decode_batch, edit_distance, read_manifest
from .model import load_checkpoint, save_checkpoint


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_steps < 0:
        raise ValueError("epochs, batch size and learning rate must be positive; max steps >= 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("use cpu or cuda; PyTorch CTC is not supported here on MPS")
    model, checkpoint = load_checkpoint(args.checkpoint)
    meta = dict(checkpoint["meta"])
    vocab = Vocab.from_json(meta["vocab"])
    rows = read_manifest(args.train)
    valid_rows = read_manifest(args.valid) if args.valid else None
    if args.resume and args.extend_vocab:
        raise ValueError("--resume cannot be combined with --extend-vocab")
    if args.extend_vocab:
        extra = sorted(set().union(*(vocab.oov(r["text"]) for r in rows)))
        if extra:
            vocab = Vocab(vocab.chars + extra)
            model.resize_classes(vocab.n_classes)
            meta["vocab"] = vocab.to_json()
            print(f"Added {len(extra)} tokens: {extra!r}", flush=True)
    train = Lines(rows, vocab, meta["rec_height"], model.config["stride"])
    valid = Lines(valid_rows, vocab, meta["rec_height"], model.config["stride"]) if valid_rows else None
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0, generator=generator)
    valid_loader = (DataLoader(valid, batch_size=args.batch_size, collate_fn=collate,
                               num_workers=0) if valid else None)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, global_step = 0, 0
    best = float("inf")
    if args.resume:
        if "optimizer" not in checkpoint:
            raise ValueError("--resume needs a training checkpoint with optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        best = checkpoint.get("best_metric", best)
        if "rng" in checkpoint:
            torch.set_rng_state(checkpoint["rng"]["torch"])
            generator.set_state(checkpoint["rng"]["loader"])
            if device == "cuda" and checkpoint["rng"].get("cuda"):
                torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        if args.max_steps and global_step >= args.max_steps:
            raise ValueError("checkpoint has already reached --max-steps; increase the limit")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)

    def batch_loss(batch):
        images, widths, targets, target_lengths, texts = batch
        logits, lengths = model(images.to(device), widths.to(device))
        loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets.to(device),
                         lengths.cpu(), target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite CTC loss; inspect line boxes and labels")
        return loss, logits, lengths, texts

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        # Only the stem retains BatchNorm. Preserve pretrained running statistics
        # for small batches and keep its affine parameters trainable.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        total_loss, count = 0.0, 0
        stopped = False
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, texts = batch_loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count += len(texts)
            total_loss += loss.item() * len(texts)
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                stopped = True
                break
        record = {"epoch": epoch + 1, "step": global_step, "train_loss": total_loss / count}
        if valid_loader:
            model.eval()
            errors, chars, valid_loss, valid_count = 0, 0, 0.0, 0
            with torch.no_grad():
                for batch in valid_loader:
                    loss, logits, lengths, texts = batch_loss(batch)
                    predictions = decode_batch(logits, lengths.cpu(), vocab)
                    errors += sum(edit_distance(t, p) for t, p in zip(texts, predictions))
                    chars += sum(len(t) for t in texts)
                    valid_count += len(texts)
                    valid_loss += loss.item() * len(texts)
            record.update(valid_loss=valid_loss / valid_count, cer=errors / max(chars, 1))
        metric = record.get("cer", record["train_loss"])
        improved = metric < best
        best = min(best, metric)
        state = {
            "epoch": epoch + 1, "step": global_step, "best_metric": best,
            "optimizer": optimizer.state_dict(), "metrics": record,
            "rng": {"torch": torch.get_rng_state(), "loader": generator.get_state(),
                    "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else []},
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
