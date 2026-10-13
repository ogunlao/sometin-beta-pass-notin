#!/usr/bin/env python3
"""Batch-transcribe audio files or directories with SBPN.

For a single file, ``python sbpn.py sample.wav`` is simpler. This script adds
directory input and TSV/JSONL output on top of the same ``SBPN`` interface.

Examples:
    python scripts/transcribe.py clips/ --model base --batch-size 8
    python scripts/transcribe.py clips/ a.wav b.wav --output results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sbpn import SBPN  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def collect_audio(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Not found: {p}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", help="Audio files and/or directories")
    parser.add_argument("--model", default="large", help="'base', 'large', a Hugging Face id or a .nemo file")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: cuda if available)")
    parser.add_argument("--output", help="Write results to a .jsonl or .tsv file instead of stdout")
    args = parser.parse_args()

    files = collect_audio(args.audio)
    if not files:
        sys.exit("No audio files found.")

    results = SBPN(args.model, args.device).transcribe(files, batch_size=args.batch_size)
    rows = [{"audio": str(f), "language": r.language, "text": r.text} for f, r in zip(files, results)]

    if args.output and args.output.endswith(".jsonl"):
        with open(args.output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        lines = ["audio\tlanguage\ttext"] + [f"{r['audio']}\t{r['language']}\t{r['text']}" for r in rows]
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            print("\n".join(lines))


if __name__ == "__main__":
    main()
