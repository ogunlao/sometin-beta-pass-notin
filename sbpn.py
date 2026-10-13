#!/usr/bin/env python3
"""Simple interface to the SBPN multilingual ASR models.

Give it a .wav file, get back the transcript and the spoken language.

Python:
    from sbpn import SBPN

    model = SBPN()                          # SBPN-Large; use SBPN("base") for the 120M model
    result = model.transcribe("sample.wav")
    result.text                             # "how body dey"
    result.language                         # "pcm"
    result.language_name                    # "Nigerian Pidgin"

    model.transcribe(["a.wav", "b.wav"])    # a list in, a list of results out

Command line:
    python sbpn.py sample.wav
    python sbpn.py sample.wav --model base
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

MODELS = {
    "base": "ogunlao/SBPN_multilingual_base",
    "large": "ogunlao/SBPN_multilingual_large",
}

# Tag emitted by the model -> (language code used in the paper, readable name).
LANGUAGES = {
    "en": ("en", "English"),
    "yo": ("yo", "Yorùbá"),
    "ha": ("ha", "Hausa"),
    "ig": ("ig", "Igbo"),
    "pd": ("pcm", "Nigerian Pidgin"),
}

SAMPLE_RATE = 16_000
_TAG_RE = re.compile(r"<\|?([A-Za-z_-]+)\|?>")


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None
    language_name: str | None

    def __str__(self) -> str:
        return f"[{self.language}] {self.text}"


def parse_output(raw: str) -> Transcription:
    """Split a raw SBPN output such as ``<|pd|> how body dey`` into text and language."""
    match = _TAG_RE.search(raw)
    tag = match.group(1) if match else None
    language, language_name = LANGUAGES.get(tag, (tag, None))
    text = " ".join(_TAG_RE.sub(" ", raw).split())
    return Transcription(text=text, language=language, language_name=language_name)


class SBPN:
    def __init__(self, model: str = "large", device: str | None = None):
        """Load an SBPN model.

        Args:
            model: "base", "large", a Hugging Face model id, or a path to a local .nemo file.
            device: e.g. "cuda", "cpu". Defaults to CUDA when available.
        """
        import torch
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from omegaconf import open_dict

        if model.endswith(".nemo"):
            load = lambda **kw: EncDecHybridRNNTCTCBPEModel.restore_from(model, **kw)  # noqa: E731
        else:
            load = lambda **kw: EncDecHybridRNNTCTCBPEModel.from_pretrained(MODELS.get(model, model), **kw)  # noqa: E731

        # The checkpoints were trained with Graph RNNT loss, which requires k2. The loss is unused at
        # inference, so swap in NeMo's default RNNT loss to load the model without installing k2.
        cfg = load(return_config=True)
        with open_dict(cfg):
            cfg.loss = {"loss_name": "default"}

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load(override_config_path=cfg, map_location=torch.device(device)).eval()

    def transcribe(
        self, audio: str | Path | list[str | Path], batch_size: int = 4
    ) -> Transcription | list[Transcription]:
        """Transcribe one audio file (returns a Transcription) or a list of files (returns a list)."""
        import librosa

        single = isinstance(audio, (str, Path))
        paths = [audio] if single else list(audio)
        # SBPN expects 16 kHz mono; resampling here means any sample rate or channel count works.
        waveforms = [librosa.load(str(p), sr=SAMPLE_RATE, mono=True)[0] for p in paths]
        outputs = self.model.transcribe(audio=waveforms, batch_size=batch_size, verbose=False)
        results = [parse_output(o.text if hasattr(o, "text") else str(o)) for o in outputs]
        return results[0] if single else results

    __call__ = transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a .wav file with SBPN and identify its language.")
    parser.add_argument("audio", help="Path to an audio file, e.g. sample.wav")
    parser.add_argument("--model", default="large", help="'base', 'large', a Hugging Face id or a .nemo file")
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu (default: cuda if available)")
    args = parser.parse_args()

    # Keep the terminal output to just the answer.
    import warnings
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel(nemo_logging.ERROR)
    warnings.filterwarnings("ignore")

    result = SBPN(args.model, args.device).transcribe(args.audio)
    print(f"Language:   {result.language_name or 'unknown'} ({result.language})")
    print(f"Transcript: {result.text}")


if __name__ == "__main__":
    main()
