"""Export a fine-tuned recognizer into a standalone Indra model directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import onnx
import torch

from .model import Inference, load_checkpoint
from .recover import compare, sha256


def run(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint)
    model.eval()
    base, out = Path(args.base_model_dir), Path(args.out)
    for name in ("det.onnx", "rec.onnx", "meta.json"):
        if not (base / name).is_file():
            raise ValueError(f"base model directory lacks {name}")
    if sha256(base / "meta.json") != checkpoint["source"]["meta_sha256"]:
        raise ValueError("base meta.json differs from the source used for reconstruction")
    if sha256(base / "rec.onnx") != checkpoint["source"]["rec_sha256"]:
        raise ValueError("base rec.onnx differs from the source used for reconstruction")
    out.mkdir(parents=True, exist_ok=False)
    x = torch.zeros(2, 1, model.config["height"], 256)
    widths = torch.tensor([127, 256], dtype=torch.int64)
    # The legacy exporter produces the same portable opset as the public model,
    # including dynamic batch and width, without an onnxscript dependency.
    torch.onnx.export(
        Inference(model).eval(), (x, widths), str(out / "rec.onnx"),
        input_names=["x", "widths"], output_names=["ids", "conf", "lengths"],
        dynamic_axes={"x": {0: "batch", 3: "width"}, "widths": {0: "batch"},
                      "ids": {0: "batch", 1: "time"}, "conf": {0: "batch", 1: "time"},
                      "lengths": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(out / "rec.onnx"))
    report = compare(model, out / "rec.onnx", args.manifest)
    # Never carry over rec.int8.onnx: OrtOCR would prefer those stale weights.
    for path in base.glob("*.onnx"):
        if path.name not in ("rec.onnx", "rec.int8.onnx"):
            shutil.copy2(path, out / path.name)
    for name in ("LICENSE", "math_meta.json"):
        if (base / name).exists():
            shutil.copy2(base / name, out / name)
    (out / "meta.json").write_text(json.dumps(checkpoint["meta"], ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    details = dict(source=checkpoint["source"], checkpoint_sha256=sha256(args.checkpoint),
                   parity=report, classes=model.config["classes"])
    (out / "finetune.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(dict(model_dir=str(out.resolve()), parity=report), indent=2))
