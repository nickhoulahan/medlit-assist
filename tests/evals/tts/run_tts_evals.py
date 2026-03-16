import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer

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


def run_tts_model(model_name: str) -> float:
    time_start = time.time()
    subprocess.run(
        ["python", f"tests/evals/tts/test_models/tts_model_{model_name}.py"],
        check=True,
    )
    time_end = time.time()
    duration = time_end - time_start
    print(f"TTS model '{model_name}' evaluation completed in {duration:.2f} seconds.")
    return duration


def evaluate_tts_output_with_asr(
    asr_model: ASRModel, tts_output_path: Path, model_name: str
) -> ASRMetrics:
    ground_truth = (
        open(Path(__file__).parent / "test_assets" / "sample_text.txt").read().strip()
    )
    transcribed_text = asr_model.transcribe(audio_input=tts_output_path)
    wer = jiwer.wer(ground_truth, transcribed_text)
    cer = jiwer.cer(ground_truth, transcribed_text)
    mer = jiwer.mer(ground_truth, transcribed_text)
    wil = jiwer.wil(ground_truth, transcribed_text)
    wip = jiwer.wip(ground_truth, transcribed_text)
    metrics = ASRMetrics(
        model=model_name,
        wer=wer,
        cer=cer,
        mer=mer,
        wil=wil,
        wip=wip,
    )
    return metrics.serialize()


if __name__ == "__main__":
    models = {
        "facebook": "facebook/mms-tts-eng",
        "piper": "piper-config-en_US-kathleen-low",
        "qwen": "qwen3-tts-12hz-0.6b-base",
    }

    durations = {}
    for model in models:
        durations[model] = run_tts_model(model_name=model)
    output_path = (
        Path(__file__).parent / "test_output" / "tts_latency.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(durations, f, indent=4)

    metrics = {}
    asr_model = ASRModel(model_name="openai/whisper-large-v3")
    for model in models:
        tts_output_path = Path(__file__).parent / "test_output" / f"tts_{model}.wav"
        metrics[model] = evaluate_tts_output_with_asr(
            asr_model, tts_output_path, model_name=models[model]
        )
    metrics_output_path = (
        Path(__file__).parent / "test_output" / "tts_asr_metrics.json"
    )
    with open(metrics_output_path, "w") as f:
        json.dump(metrics, f, indent=4)
