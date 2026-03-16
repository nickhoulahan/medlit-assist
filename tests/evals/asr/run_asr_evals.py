import json
from dataclasses import dataclass
from pathlib import Path

import jiwer
from slugify import slugify

from src.asr.asr_model import ASRModel


@dataclass
class ASRMetrics:
    model: str
    wer: float
    cer: float
    mer: float
    wil: float
    wip: float

    def serialize(self) -> dict:
        return {
            "model": self.model,
            "wer": self.wer,
            "cer": self.cer,
            "mer": self.mer,
            "wil": self.wil,
            "wip": self.wip,
        }


def get_asr_model(model_name):
    asr_model = ASRModel(model_name=model_name)
    return asr_model


def load_reference_transcripts(transcript_dir):
    transcripts = {}
    for txt_file in transcript_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            transcripts[txt_file.stem] = f.read().strip()
    return transcripts


def evaluate_metrics(asr_model, audio_dir, transcript_dir):
    references = load_reference_transcripts(transcript_dir)
    for audio_file in audio_dir.glob("*.wav"):
        reference = references.get(audio_file.stem)
        hypothesis = asr_model.transcribe(audio_input=audio_file)
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        mer = jiwer.mer(reference, hypothesis)
        wil = jiwer.wil(reference, hypothesis)
        wip = jiwer.wip(reference, hypothesis)
        metrics = ASRMetrics(
            model=asr_model.model_name,
            wer=wer,
            cer=cer,
            mer=mer,
            wil=wil,
            wip=wip,
        )
        with open(
            Path(__file__).parent
            / "reports"
            / f"{audio_file.stem}_{slugify(asr_model.model_name)}_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics.serialize(), f, indent=4)


def main():
    candidate_models = [
        "openai/whisper-small",
        "openai/whisper-large-v3",
        "ibm-granite/granite-4.0-1b-speech",
    ]
    audio_dir = Path(__file__).parent / "eval_resources"
    transcript_dir = Path(__file__).parent / "eval_resources"
    for model_name in candidate_models:
        asr_model = get_asr_model(model_name)
        evaluate_metrics(asr_model, audio_dir, transcript_dir)


if __name__ == "__main__":
    main()
