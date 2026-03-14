from pathlib import Path
from types import SimpleNamespace

import torch

from src.asr.asr_model import ASRModel


class _FakeLoadedModel:
    def __init__(self):
        self.to_device = None

    def to(self, device: str = None, **_kwargs):
        self.to_device = device
        return self

# mock the transformers pipeline
def test_init_builds_cpu_pipeline(monkeypatch):
    fake_model = _FakeLoadedModel()
    fake_processor = SimpleNamespace(
        tokenizer="fake-tokenizer", feature_extractor="fake-feature-extractor"
    )
    called = {"model": None, "processor": None, "pipe": None}

    def fake_model_from_pretrained(*args, **kwargs):
        called["model"] = {"args": args, "kwargs": kwargs}
        return fake_model

    def fake_processor_from_pretrained(*args, **kwargs):
        called["processor"] = {"args": args, "kwargs": kwargs}
        return fake_processor

    def fake_pipeline(**kwargs):
        called["pipe"] = kwargs
        return lambda _audio_input, generate_kwargs=None: {"text": "ok"}

    monkeypatch.setattr(
        "src.asr.asr_model.AutoModelForSpeechSeq2Seq.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "src.asr.asr_model.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr("src.asr.asr_model.pipeline", fake_pipeline)

    model = ASRModel(model_name="fake-model")

    assert model.torch_dtype == torch.float32
    assert fake_model.to_device == "cpu"
    assert called["model"]["args"] == ("fake-model",)
    assert called["processor"]["args"] == ("fake-model",)
    assert called["pipe"]["task"] == "automatic-speech-recognition"
    assert called["pipe"]["dtype"] == torch.float32
    assert called["pipe"]["device"] == -1


def test_transcribe_returns_clean_text_and_passes_kwargs():
    model = object.__new__(ASRModel)
    called = {"audio": None, "kwargs": None}

    def fake_pipeline(audio_input, generate_kwargs=None):
        called["audio"] = audio_input
        called["kwargs"] = generate_kwargs
        return {"text": " test words "}

    model.asr_pipeline = fake_pipeline

    result = model.transcribe(Path("sample.wav"), generate_kwargs={"language": "en"})

    assert result == "test words"
    assert called["audio"] == "sample.wav"
    assert called["kwargs"] == {"language": "en"}


def test_generate_text_response_delegates_to_transcribe():
    model = object.__new__(ASRModel)
    called = {"audio": None, "kwargs": None}

    def fake_transcribe(audio_input, generate_kwargs=None):
        called["audio"] = audio_input
        called["kwargs"] = generate_kwargs
        return "passed"

    model.transcribe = fake_transcribe

    result = model.generate_text_response(
        Path("audio.wav"), generate_kwargs={"task": "transcribe"}
    )

    assert result == "passed"
    assert called == {"audio": Path("audio.wav"), "kwargs": {"task": "transcribe"}}
