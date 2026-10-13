#!/usr/bin/env python3
"""Transcribe audio with an SBPN model and report the predicted language.

SBPN models prepend a language tag to every transcript, e.g.
``<|pd|> how body dey``. This script splits that tag off so you get both the
clean transcript and the language identification (LID) prediction.

Examples:
    python scripts/transcribe.py sample.wav
    python scripts/transcribe.py clips/ --model large --batch-size 8 --output out.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

MODELS = {
    "base": "ogunlao/SBPN_multilingual_base",
    "large": "ogunlao/SBPN_multilingual_large",
}

# Tag emitted by the model -> ISO 639 code used in the paper.
LANG_TAGS = {"en": "en", "yo": "yo", "ha": "ha", "ig": "ig", "pd": "pcm"}

TAG_RE = re.compile(r"<\|?([A-Za-z_-]+)\|?>")
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
SAMPLE_RATE = 16_000


def split_language_tag(text: str) -> tuple[str | None, str]:
    """Return (language, transcript) for a raw SBPN output string."""
    match = TAG_RE.search(text)
    lang = LANG_TAGS.get(match.group(1), match.group(1)) if match else None
    clean = " ".join(TAG_RE.sub(" ", text).split())
    return lang, clean


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
    parser.add_argument(
        "--model",
        default="large",
        help="'base', 'large', a Hugging Face model id, or a local .nemo file (default: large)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: cuda if available)")
    parser.add_argument("--output", help="Write results to a .jsonl or .tsv file instead of stdout")
    args = parser.parse_args()

    import librosa
    import torch
    import nemo.collections.asr as nemo_asr

    files = collect_audio(args.audio)
    if not files:
        sys.exit("No audio files found.")

    model_cls = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    if args.model.endswith(".nemo"):
        model = model_cls.restore_from(args.model)
    else:
        model = model_cls.from_pretrained(model_name=MODELS.get(args.model, args.model))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # SBPN expects 16 kHz mono; resample everything up front so any input format works.
    audio = [librosa.load(str(f), sr=SAMPLE_RATE, mono=True)[0] for f in files]
    outputs = model.transcribe(audio=audio, batch_size=args.batch_size)

    results = []
    for path, hyp in zip(files, outputs):
        raw = hyp.text if hasattr(hyp, "text") else str(hyp)
        lang, text = split_language_tag(raw)
        results.append({"audio": str(path), "language": lang, "text": text})

    if args.output and args.output.endswith(".jsonl"):
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        lines = ["audio\tlanguage\ttext"] + [f"{r['audio']}\t{r['language']}\t{r['text']}" for r in results]
        if args.output:
            Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            print("\n".join(lines))


if __name__ == "__main__":
    main()
