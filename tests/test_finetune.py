"""No-network checks; released-weight integration tests skip without a local model set."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from indra_ocr.finetune.data import Lines, read_manifest
from indra_ocr.finetune.model import Recognizer, load_checkpoint, save_checkpoint
from indra_ocr.finetune.recover import compare, recover, sha256
from indra_ocr.finetune.train import run as train
from indra_ocr.finetune.export import run as export
from indra_ocr.pipeline import OrtOCR
from indra_ocr.rec_preproc import crop_line, pad_batch
from indra_ocr.vocab import Vocab

SOURCE = Path(os.environ.get("INDRA_FINETUNE_SOURCE", ".model-source"))
release = pytest.mark.skipif(not (SOURCE / "rec.onnx").exists(),
                             reason="download the released model set for integration tests")


def manifest(tmp_path, texts):
    rows = []
    for i, text in enumerate(texts):
        image = np.full((40, 320), 255, np.uint8)
        cv2.putText(image, "HELLO WORLD", (2, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 1)
        cv2.imwrite(str(tmp_path / f"{i}.png"), image)
        rows.append({"image": f"{i}.png", "text": text})
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_invalid_labels_are_rejected_before_ctc(tmp_path):
    path = manifest(tmp_path, ["aaa"])
    rows = read_manifest(path)
    with pytest.raises(ValueError, match="unknown tokens"):
        Lines(rows, Vocab(["b"]), 32)
    rows[0]["bbox"] = [0, 0, 2, 40]
    dataset = Lines(rows, Vocab(["a"]), 32)
    with pytest.raises(ValueError, match="CTC needs 5 steps"):
        dataset[0]


def test_vocabulary_extension_preserves_pretrained_classes():
    torch.manual_seed(0)
    model = Recognizer(classes=3)
    old_weight, old_bias = model.head.weight.clone(), model.head.bias.clone()
    model.resize_classes(4)
    torch.testing.assert_close(model.head.weight[:3], old_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.head.bias[:3], old_bias, rtol=0, atol=0)
    assert model.config["classes"] == 4


@release
def test_released_weights_match_onnx():
    torch.set_num_threads(2)
    model, _ = recover(SOURCE)
    report = compare(model, SOURCE / "rec.onnx")
    assert report["lines"] == 11
    assert report["ids_equal"] and report["lengths_equal"]


@release
def test_train_resume_extend_export_and_runtime(tmp_path):
    torch.set_num_threads(2)
    model, meta = recover(SOURCE)
    initial_weight = model.head.weight.detach().clone()
    initial_classes = model.config["classes"]
    checkpoint = tmp_path / "recovered.pt"
    provenance = {"rec_sha256": sha256(SOURCE / "rec.onnx"),
                  "meta_sha256": sha256(SOURCE / "meta.json")}
    save_checkpoint(checkpoint, model, meta, provenance)
    # A new token exercises actual CTC gradients through an extended head.
    labels = manifest(tmp_path, ["HELLO𐀀", "WORLD𐀀"])
    args = SimpleNamespace(checkpoint=str(checkpoint), train=str(labels), valid=str(labels),
                           out=str(tmp_path / "train"), epochs=1, batch_size=2, lr=1e-4,
                           device="cpu", extend_vocab=True, resume=False, max_steps=1,
                           seed=42, threads=2, weight_decay=1e-4)
    train(args)
    trained, state = load_checkpoint(tmp_path / "train" / "last.pt")
    assert state["step"] == 1
    assert trained.config["classes"] == initial_classes + 1
    assert not torch.equal(trained.head.weight[:initial_classes], initial_weight)
    assert torch.isfinite(torch.tensor(state["metrics"]["valid_loss"]))
    assert "𐀀" in Vocab.from_json(state["meta"]["vocab"]).index
    args.checkpoint = str(tmp_path / "train" / "last.pt")
    args.resume, args.extend_vocab, args.max_steps = True, False, 2
    args.out = str(tmp_path / "resumed")
    train(args)
    trained, state = load_checkpoint(tmp_path / "resumed" / "last.pt")
    assert state["step"] == 2 and state["epoch"] == 2
    output = tmp_path / "exported"
    export(SimpleNamespace(checkpoint=str(tmp_path / "resumed" / "last.pt"),
                           base_model_dir=str(SOURCE), out=str(output), threads=2,
                           manifest=str(labels)))
    assert not (output / "rec.int8.onnx").exists()
    assert Vocab.from_json(json.loads((output / "meta.json").read_text())["vocab"]).n_classes == initial_classes + 1
    # Exercise the actual public runtime loader and its three-output contract.
    runtime = OrtOCR(output, threads=2)
    gray = cv2.imread(str(tmp_path / "0.png"), cv2.IMREAD_GRAYSCALE)
    boxes = np.array([[0, 0, gray.shape[1], gray.shape[0]]], np.float32)
    result = runtime.recognize(gray, boxes)
    assert len(result) == 1
    assert isinstance(result[0][0], str)
    trained.eval()
    arr, widths = pad_batch([crop_line(gray, boxes[0], meta["rec_height"])])
    with torch.no_grad():
        logits, lengths = trained(torch.from_numpy(arr), torch.from_numpy(widths))
    ids, confidence, actual_lengths = runtime.rec.run(None, {"x": arr, "widths": widths})
    np.testing.assert_array_equal(ids, logits.argmax(-1).numpy())
    np.testing.assert_array_equal(actual_lengths, lengths.numpy())
    np.testing.assert_allclose(confidence, logits.softmax(-1).amax(-1).numpy(), atol=2e-5, rtol=1e-4)
