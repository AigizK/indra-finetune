"""python -m indra_ocr.finetune {download,recover,verify,train,export}."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download a pinned public model set")
    download.add_argument("--out", required=True)
    download.add_argument("--revision", default="v3.1.1")
    download.add_argument("--with-safetensors", action="store_true")
    download.add_argument("--math", action="store_true")
    recover = sub.add_parser("recover", help="restore PyTorch weights; require ONNX parity")
    recover.add_argument("--source", required=True)
    recover.add_argument("--out", required=True)
    recover.add_argument("--manifest", help="optional labelled real images for additional parity checks")
    verify = sub.add_parser("verify", help="compare a restored checkpoint with its source ONNX")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--onnx", required=True)
    verify.add_argument("--manifest")
    train = sub.add_parser("train", help="fine-tune the line recognizer with CTC")
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--train", required=True, help="JSONL manifest")
    train.add_argument("--valid", help="independent validation JSONL manifest")
    train.add_argument("--out", required=True, help="new run directory")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--extend-vocab", action="store_true")
    train.add_argument("--resume", action="store_true", help="also restore optimizer and epoch counter")
    train.add_argument("--max-steps", type=int, default=0, help="stop at this total step count; 0 = unlimited")
    train.add_argument("--seed", type=int, default=42)
    export = sub.add_parser("export", help="export a compatible Indra model directory")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--base-model-dir", required=True)
    export.add_argument("--out", required=True, help="new directory")
    export.add_argument("--manifest")
    for command in (recover, verify, train, export):
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if getattr(args, "threads", 1) < 1:
        parser.error("--threads must be >= 1")
    if args.command == "download":
        from huggingface_hub import snapshot_download
        patterns = ["*.onnx", "meta.json", "LICENSE"]
        if args.with_safetensors:
            patterns.append("model.safetensors")
        if args.math:
            patterns.append("math_meta.json")
        print(snapshot_download("datalab-to/indra-ocr", revision=args.revision, local_dir=args.out,
                                allow_patterns=patterns,
                                ignore_patterns=[] if args.math else ["math*.onnx"]))
    elif args.command == "recover":
        from .recover import run_recover
        run_recover(args)
    elif args.command == "verify":
        from .recover import run_verify
        run_verify(args)
    elif args.command == "train":
        from .train import run
        run(args)
    elif args.command == "export":
        from .export import run
        run(args)


if __name__ == "__main__":
    main()
