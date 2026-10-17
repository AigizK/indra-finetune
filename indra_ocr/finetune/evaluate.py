"""End-to-end page evaluation with identical settings for baseline and candidate."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .cache import file_hash
from .data import read_manifest
from .metrics import TranscriptionMetrics


def page_text(converter, row):
    number = row.get("page")
    if number is not None and (type(number) is not int or number < 0):
        raise ValueError("page must be a zero-based nonnegative integer")
    if number is None:
        pages = list(converter.convert(row["image"], max_pages=2))
        if len(pages) != 1:
            raise ValueError("multi-page inputs need an explicit zero-based page index")
        _, page = pages[0]
    else:
        page = next((p for i, p in converter.convert(row["image"], max_pages=number + 1)
                     if i == number), None)
        if page is None:
            raise ValueError(f"page {number} does not exist")
    if page.source == "error":
        raise ValueError(page.note or "OCR failed")
    return page.text or "", {"source": page.source, "rotation": page.rotation,
                             "detected_lines": len(page.lines or [])}


def score_pages(converter, rows, output, normalization):
    aggregate = TranscriptionMetrics(normalization)
    failures, seconds = 0, 0.0
    with Path(output).open("w", encoding="utf-8") as stream:
        for row in rows:
            started = time.perf_counter()
            error, metadata = None, {}
            try:
                prediction, metadata = page_text(converter, row)
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
            elapsed = time.perf_counter() - started
            seconds += elapsed
            metrics = aggregate.add(row["text"], prediction)
            stream.write(json.dumps({"image": row["image"], "page": row.get("page"),
                                     "reference": row["text"], "prediction": prediction,
                                     "metrics": metrics, "error": error, "seconds": elapsed,
                                     **metadata}, ensure_ascii=False) + "\n")
    return dict(aggregate.result(), failed_pages=failures, seconds=seconds)


def run(args):
    import indra_ocr
    rows = read_manifest(args.manifest)
    for row in rows:
        if "bbox" in row:
            raise ValueError("page evaluation expects whole-page text without bbox")
        if "page" in row and (type(row["page"]) is not int or row["page"] < 0):
            raise ValueError(f"{row['location']}: page must be a nonnegative integer")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    models = {"baseline": args.model_dir}
    if args.compare_model_dir:
        models["candidate"] = args.compare_model_dir
    report = {"manifest_sha256": file_hash(args.manifest),
              "settings": {"mode": args.mode, "dpi": args.dpi, "threads": args.threads,
                           "furniture": True, "normalization": args.normalization,
                           "text": "page.text; preserve case, punctuation and inner whitespace"},
              "models": {}}
    for name, directory in models.items():
        directory = Path(directory).resolve()
        converter = indra_ocr.load(path=directory, threads=args.threads, dpi=args.dpi,
                                   mode=args.mode, furniture=True,
                                   providers=["CPUExecutionProvider"])
        result = score_pages(converter, rows, out / f"{name}.jsonl", args.normalization)
        result["model_dir"] = str(directory)
        result["model_sha256"] = {p.name: file_hash(p) for p in directory.iterdir()
                                  if p.suffix == ".onnx" or p.name == "meta.json"}
        report["models"][name] = result
        print(name + ": " + json.dumps(result), flush=True)
        del converter
    if "candidate" in report["models"]:
        baseline, candidate = (report["models"][k] for k in ("baseline", "candidate"))
        report["delta"] = {metric: candidate[metric] - baseline[metric] for metric in ("cer", "wer")}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Every failed page remains in the denominator and in the predictions. A
    # partially failed evaluation must also fail a CLI/CI invocation.
    if any(m["failed_pages"] for m in report["models"].values()):
        raise RuntimeError(f"page evaluation had failures; see {out / 'report.json'}")
    return report
