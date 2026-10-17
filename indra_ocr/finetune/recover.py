"""Recover native trainable layers and require numerical parity with ONNX."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from indra_ocr.rec_preproc import crop_line, pad_batch
from indra_ocr.vocab import Vocab
from .data import read_manifest
from .model import Inference, Recognizer, load_checkpoint, save_checkpoint


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def recover(source: Path):
    graph = onnx.load(source / "rec.onnx")
    weights = {t.name: numpy_helper.to_array(t).copy() for t in graph.graph.initializer}
    conv = {n.name: n for n in graph.graph.node if n.op_type == "Conv"}
    stem = conv["/rec/stem/Conv"]
    stem_shape = weights[stem.input[1]].shape
    channels, in_channels, height, stride = stem_shape
    if in_channels != 1 or attrs(stem).get("strides") != [height, stride]:
        raise ValueError("unsupported stem geometry")
    names = sorted(int(re.fullmatch(r"/rec/blocks\.(\d+)/dw/Conv", name)[1])
                   for name in conv if re.fullmatch(r"/rec/blocks\.(\d+)/dw/Conv", name))
    if not names or names != list(range(len(names))):
        raise ValueError("unsupported residual block layout")
    bn = next(n for n in graph.graph.node if n.name == "/rec/stem_bn/BatchNormalization")
    classes = weights["rec.head.weight"].shape[0]
    expansion = weights["rec.blocks.0.expand.weight"].shape[0]
    se_hidden = weights["rec.blocks.0.se.fc1.weight"].shape[0]
    kernel = weights[conv["/rec/blocks.0/dw/Conv"].input[1]].shape[-1]
    model = Recognizer(classes, height, channels, expansion, se_hidden, len(names), kernel,
                       stride, float(attrs(bn)["epsilon"]))
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    vocab = Vocab.from_json(meta["vocab"])
    if vocab.n_classes != classes or meta["rec_height"] != height:
        raise ValueError("meta.json does not match the recognizer")
    if vocab.n_classes != len(vocab.index) + 1:
        raise ValueError("duplicate vocabulary tokens")
    # Safetensors uses ONNX initializer names, including fused Conv weights.
    safe_path = source / "model.safetensors"
    if safe_path.exists():
        from safetensors.numpy import load_file
        safe = {name.removeprefix("rec/"): value for name, value in load_file(safe_path).items()}
        if safe.keys() != weights.keys() or any(
            not np.array_equal(safe[name], value) for name, value in weights.items()
        ):
            raise ValueError("model.safetensors and rec.onnx contain different weights/versions")
        weights = safe
    state = model.state_dict()
    for name, value in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        match = re.fullmatch(r"blocks\.(\d+)\.dw\.(weight|bias)", name)
        if match:
            i, kind = match.groups()
            node = conv[f"/rec/blocks.{i}/dw/Conv"]
            expected = {"dilations": [1], "group": channels, "kernel_shape": [kernel],
                        "pads": [kernel // 2, kernel // 2], "strides": [1]}
            if attrs(node) != expected:
                raise ValueError(f"unsupported depthwise convolution: {node.name}")
            key = node.input[1 if kind == "weight" else 2]
        else:
            key = "rec." + name
        array = weights[key]
        if array.shape != tuple(value.shape) or not np.isfinite(array).all():
            raise ValueError(f"invalid weight: {key}")
        state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)
    return model.eval(), meta


def compare(model, onnx_path, manifest=None, limit=32):
    """Compare raw logits as well as the exported ids, confidence and lengths."""
    graph = onnx.load(onnx_path)
    # Recover logits before ArgMax; the public runtime graph only returns three
    # reduced outputs. Merely comparing recognized text could hide bad weights.
    argmax = next(n for n in graph.graph.node if n.op_type == "ArgMax")
    logits_name = argmax.input[0]
    original_names = [o.name for o in graph.graph.output]
    graph.graph.output.append(helper.make_tensor_value_info(
        logits_name, onnx.TensorProto.FLOAT, [None, None, model.config["classes"]]
    ))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(graph.SerializeToString(), opts,
                                  providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    batches = []
    for widths in ([8], [9, 31, 64], [127, 256], [511, 777], [2048, 4096]):
        crops = [rng.random((model.config["height"], w), dtype=np.float32) for w in widths]
        batches.append(pad_batch(crops))
    batches.append(pad_batch([np.zeros((model.config["height"], 80), dtype=np.float32)]))
    if manifest:
        import cv2
        rows = read_manifest(manifest)[:limit]
        for start in range(0, len(rows), 4):
            crops = []
            for row in rows[start:start + 4]:
                gray = cv2.imread(row["image"], cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError(f"cannot decode {row['image']}")
                h, w = gray.shape
                crops.append(crop_line(gray, row.get("bbox", [0, 0, w, h]), model.config["height"]))
            batches.append(pad_batch(crops))
    report = {"batches": len(batches), "lines": 0, "max_logit_error": 0.0,
              "max_confidence_error": 0.0, "ids_equal": True, "lengths_equal": True}
    model.eval()
    wrapper = Inference(model).eval()
    with torch.no_grad():
        for x, widths in batches:
            reference = session.run(original_names + [logits_name], {"x": x, "widths": widths})
            logits, _ = model(torch.from_numpy(x), torch.from_numpy(widths))
            actual = [v.numpy() for v in wrapper(torch.from_numpy(x), torch.from_numpy(widths))]
            error = float(np.max(np.abs(logits.numpy() - reference[-1])))
            report["max_logit_error"] = max(report["max_logit_error"], error)
            report["max_confidence_error"] = max(report["max_confidence_error"],
                                                float(np.max(np.abs(actual[1] - reference[1]))))
            report["lines"] += len(widths)
            np.testing.assert_allclose(logits.numpy(), reference[-1], atol=5e-4, rtol=1e-4)
            np.testing.assert_array_equal(actual[0], reference[0])
            np.testing.assert_allclose(actual[1], reference[1], atol=2e-5, rtol=1e-4)
            np.testing.assert_array_equal(actual[2], reference[2])
    return report


def run_recover(args):
    torch.set_num_threads(args.threads)
    source = Path(args.source)
    if Path(args.out).exists():
        raise ValueError(f"output already exists: {args.out}")
    model, meta = recover(source)
    report = compare(model, source / "rec.onnx", args.manifest)
    provenance = dict(rec_sha256=sha256(source / "rec.onnx"),
                      meta_sha256=sha256(source / "meta.json"),
                      repository="https://github.com/datalab-to/indra",
                      model_repository="datalab-to/indra-ocr", reconstruction="fused-inference-graph")
    if (source / "model.safetensors").exists():
        provenance["safetensors_sha256"] = sha256(source / "model.safetensors")
    save_checkpoint(args.out, model, meta, provenance, parity=report)
    print(json.dumps(dict(checkpoint=args.out, parameters=sum(p.numel() for p in model.parameters()),
                          config=model.config, parity=report), indent=2))


def run_verify(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint)
    if sha256(args.onnx) != checkpoint["source"]["rec_sha256"]:
        raise ValueError("ONNX SHA256 does not match the checkpoint's original source")
    print(json.dumps(compare(model.eval(), args.onnx, args.manifest), indent=2))
