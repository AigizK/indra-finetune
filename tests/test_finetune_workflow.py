"""Cache invalidation, batching, real resume, MPS gradients and page-level scoring."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from indra_ocr.finetune.data import Lines, WidthBatchSampler, read_manifest
from indra_ocr.finetune.evaluate import page_text, run as evaluate_pages, score_pages
from indra_ocr.finetune.metrics import TranscriptionMetrics
from indra_ocr.finetune.model import load_checkpoint
from indra_ocr.finetune.train import ctc_loss, run as train, select_device
from indra_ocr.vocab import Vocab
from test_finetune import SOURCE, release, manifest


def test_cache_matches_live_crops_and_does_not_redecode(tmp_path, monkeypatch):
    labels = manifest(tmp_path, ["HELLO", "WORLD"])
    rows = read_manifest(labels)
    vocab = Vocab(list("HELOWRD"))
    live = Lines(rows, vocab, 32)
    cached = Lines(rows, vocab, 32, cache_dir=tmp_path / "cache")
    for a, b in zip(live, cached):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert live.signature == cached.signature
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: pytest.fail("cache hit decoded an image"))
    again = Lines(rows, vocab, 32, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(again[0][0], cached[0][0])


def test_cache_invalidation_tracks_images_boxes_and_preprocessing(tmp_path):
    rows = read_manifest(manifest(tmp_path, ["HELLO"]))
    vocab = Vocab(list("HELO"))
    first = Lines(rows, vocab, 32, cache_dir=tmp_path / "cache")
    gray = cv2.imread(rows[0]["image"], 0)
    gray[0, 0] = 0
    cv2.imwrite(rows[0]["image"], gray)
    changed = Lines(rows, vocab, 32, cache_dir=tmp_path / "cache")
    assert first.cached[0][0] != changed.cached[0][0]
    assert first.signature != changed.signature
    rows[0]["bbox"] = [0, 0, 150, 40]
    cropped = Lines(rows, vocab, 32, cache_dir=tmp_path / "cache")
    assert changed.cached[0][0] != cropped.cached[0][0]
    resized = Lines(rows, vocab, 16, cache_dir=tmp_path / "cache")
    assert cropped.cached[0][0] != resized.cached[0][0]
    assert resized[0][0].shape[0] == 16
    rows[0]["text"] = "HOLE"
    relabelled = Lines(rows, vocab, 16, cache_dir=tmp_path / "cache")
    assert relabelled.cached == resized.cached  # image cache is independent of labels
    assert relabelled.signature != resized.signature


def test_width_sampler_covers_every_row_and_reduces_padding():
    widths = [64, 2048, 72, 1900, 80, 2200, 512, 560, 520, 88]
    generator = torch.Generator().manual_seed(7)
    sampler = WidthBatchSampler(widths, 3, generator, max_area=6600)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(len(widths)))
    assert all(len(b) <= 3 and max(widths[i] for i in b) * len(b) <= 6600 for b in batches)
    for b in batches:
        assert max(max(widths[i], 64) for i in b) <= 1.3 * min(max(widths[i], 64) for i in b)
    ordinary = [list(range(i, min(i + 3, len(widths)))) for i in range(0, len(widths), 3)]
    padded = lambda parts: sum(max(widths[i] for i in b) * len(b) for b in parts)
    assert padded(batches) < padded(ordinary)
    clone = WidthBatchSampler(widths, 3, torch.Generator().manual_seed(7), max_area=6600)
    assert list(clone) == batches
    odd_widths = [3149] * 13
    odd_batches = list(WidthBatchSampler(odd_widths, 13, torch.Generator(), max_area=40960))
    assert all(3152 * len(b) <= 40960 for b in odd_batches)


def test_transcription_metrics_micro_averaging_and_empty_references():
    metrics = TranscriptionMetrics("NFC")
    metrics.add("и\u0306", "й")
    metrics.add("one two", "one too")
    result = metrics.result()
    assert result["char_errors"] == 1 and result["reference_chars"] == 8
    assert result["cer"] == 1 / 8 and result["wer"] == 1 / 3
    blank = TranscriptionMetrics()
    blank.add("", "x")
    assert blank.result()["cer"] == 1
    with pytest.raises(ValueError, match="zero samples"):
        TranscriptionMetrics().result()


def test_page_failures_remain_in_metrics_and_predictions(tmp_path):
    class Converter:
        def convert(self, image, **kwargs):
            if image == "bad":
                raise ValueError("unreadable")
            yield 0, SimpleNamespace(source="ocr", text="й", rotation=0, lines=[])
    rows = [{"image": "good", "text": "и\u0306"}, {"image": "bad", "text": "abc"}]
    path = tmp_path / "predictions.jsonl"
    result = score_pages(Converter(), rows, path, "NFC")
    assert result["samples"] == 2 and result["failed_pages"] == 1
    assert result["cer"] == 3 / 4
    predictions = [json.loads(line) for line in path.read_text().splitlines()]
    assert predictions[1]["error"] and predictions[1]["prediction"] == ""


def test_multi_page_input_requires_explicit_page():
    class Converter:
        def convert(self, image, **kwargs):
            for i in range(2):
                yield i, SimpleNamespace(source="ocr", text=str(i), rotation=0, lines=[])
    with pytest.raises(ValueError, match="explicit"):
        page_text(Converter(), {"image": "two-pages"})
    assert page_text(Converter(), {"image": "two-pages", "page": 1})[0] == "1"


def test_failed_page_run_writes_report_and_raises(tmp_path, monkeypatch):
    import indra_ocr
    class BrokenConverter:
        def convert(self, *args, **kwargs):
            raise ValueError("broken image")
    monkeypatch.setattr(indra_ocr, "load", lambda **kwargs: BrokenConverter())
    labels = manifest(tmp_path, ["abc"])
    with pytest.raises(RuntimeError, match="evaluation had failures"):
        evaluate_pages(SimpleNamespace(manifest=str(labels), model_dir=str(tmp_path),
                       compare_model_dir=None, out=str(tmp_path / "report"),
                       mode="ocr", normalization="NFC", threads=2, dpi=200))
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert result["models"]["baseline"]["failed_pages"] == 1
    assert result["models"]["baseline"]["cer"] == 1


def training_args(checkpoint, labels, out, **kwargs):
    return SimpleNamespace(checkpoint=str(checkpoint), train=str(labels), valid=str(labels),
                           out=str(out), epochs=kwargs.get("epochs", 1), batch_size=2, lr=1e-4,
                           device=kwargs.get("device", "cpu"), extend_vocab=False,
                           resume=kwargs.get("resume", False), max_steps=0, seed=42,
                           threads=2, weight_decay=1e-4, cache_dir=str(out.parent / "cache"))


def test_perfect_baseline_still_produces_best_checkpoint(tmp_path):
    from indra_ocr.finetune.model import Recognizer, save_checkpoint
    model = Recognizer(classes=2, channels=4, expansion=8, se_hidden=2, blocks=1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[1] = 5
    vocab = Vocab(["A"])
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, model, {"vocab": vocab.to_json(), "rec_height": 32}, {})
    labels = manifest(tmp_path, ["A"])
    train(training_args(checkpoint, labels, tmp_path / "perfect"))
    _, state = load_checkpoint(tmp_path / "perfect/best.pt")
    assert state["baseline"]["cer"] == 0
    assert not state["metrics"]["improved_over_baseline"]


@release
def test_baseline_and_resume_match_uninterrupted_training(tmp_path):
    # Use an existing recovered checkpoint if available; otherwise recover inside
    # the test so CI has exactly the same coverage as the local machine.
    from indra_ocr.finetune.recover import recover, sha256
    from indra_ocr.finetune.model import save_checkpoint
    model, meta = recover(SOURCE)
    checkpoint = tmp_path / "initial.pt"
    save_checkpoint(checkpoint, model, meta, {"rec_sha256": sha256(SOURCE / "rec.onnx"),
                                            "meta_sha256": sha256(SOURCE / "meta.json")})
    labels = manifest(tmp_path, ["HELLO", "WORLD", "HELLO"])
    train(training_args(checkpoint, labels, tmp_path / "continuous", epochs=2))
    train(training_args(checkpoint, labels, tmp_path / "first"))
    before = json.loads((tmp_path / "first/baseline.json").read_text())
    assert before["reference"]["samples"] == 3
    first, first_state = load_checkpoint(tmp_path / "first/last.pt")
    # Baseline was evaluated before any update, and best.pt always exists.
    assert (tmp_path / "first/best.pt").is_file()
    assert first_state["baseline"] == before["reference"]
    train(training_args(tmp_path / "first/last.pt", labels, tmp_path / "resumed", resume=True))
    continuous, cstate = load_checkpoint(tmp_path / "continuous/last.pt")
    resumed, rstate = load_checkpoint(tmp_path / "resumed/last.pt")
    assert cstate["step"] == rstate["step"] and rstate["epoch"] == 2
    for k, value in continuous.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[k], atol=0, rtol=0)
    assert rstate["baseline"] == first_state["baseline"]
    assert (tmp_path / "resumed/best.pt").is_file()
    labels.write_text(labels.read_text().replace("WORLD", "HELLO"))
    with pytest.raises(ValueError, match="data changed"):
        train(training_args(tmp_path / "first/last.pt", labels, tmp_path / "changed", resume=True))


@release
def test_full_page_evaluation_runs_public_ocr_pipeline(tmp_path):
    image = np.full((400, 900), 255, np.uint8)
    cv2.putText(image, "HELLO WORLD", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.putText(image, "INDRA OCR", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.imwrite(str(tmp_path / "page.png"), image)
    labels = tmp_path / "pages.jsonl"
    labels.write_text(json.dumps({"image": "page.png", "text": "HELLO WORLD\nINDRA OCR"}) + "\n")
    report = evaluate_pages(SimpleNamespace(manifest=str(labels), model_dir=str(SOURCE),
                            compare_model_dir=str(SOURCE), out=str(tmp_path / "evaluation"),
                            mode="ocr", normalization="NFC", threads=2, dpi=200))
    assert report["models"]["baseline"]["failed_pages"] == 0
    assert report["models"]["baseline"]["cer"] == 0
    assert report["delta"] == {"cer": 0.0, "wer": 0.0}
    assert (tmp_path / "evaluation/candidate.jsonl").is_file()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_ctc_backward_and_cpu_checkpoint(tmp_path):
    from indra_ocr.finetune.model import Recognizer, save_checkpoint
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cpu = Recognizer(classes=3, blocks=1).eval()
    mps = Recognizer(classes=3, blocks=1).eval()
    mps.load_state_dict(cpu.state_dict())
    mps.to("mps")
    images = torch.rand(2, 1, 32, 64)
    widths = torch.tensor([40, 64])
    targets, target_lengths = torch.tensor([1, 2, 2, 1]), torch.tensor([2, 2])
    criterion = torch.nn.CTCLoss(blank=0)
    logits_cpu, lengths_cpu = cpu(images, widths)
    logits_mps, lengths_mps = mps(images.to("mps"), widths.to("mps"))
    a = ctc_loss(logits_cpu, lengths_cpu, targets, target_lengths, criterion)
    b = ctc_loss(logits_mps, lengths_mps, targets, target_lengths, criterion)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    a.backward(); b.backward()
    torch.testing.assert_close(cpu.head.weight.grad, mps.head.weight.grad.cpu(), rtol=2e-3, atol=1e-4)
    opt = torch.optim.AdamW(mps.parameters(), lr=1e-4)
    old = mps.head.weight.detach().cpu().clone()
    opt.step()
    assert not torch.equal(old, mps.head.weight.cpu())
    save_checkpoint(tmp_path / "mps.pt", mps, {}, {}, optimizer=opt.state_dict())
    restored, state = load_checkpoint(tmp_path / "mps.pt")
    assert all(p.device.type == "cpu" for p in restored.parameters())
    assert all(torch.isfinite(p).all() for p in restored.parameters())
