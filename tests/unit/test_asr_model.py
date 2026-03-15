from pathlib import Path
from types import SimpleNamespace

import torch

from src.asr.asr_model import ASRModel


class _FakeLoadedModel:
    def __init__(self):
        self.to_device = None
        self.generated = None

    def to(self, device: str = None, **_kwargs):
        self.to_device = device
        return self

    def generate(self, *args, **kwargs):
        self.generated = {"args": args, "kwargs": kwargs}
        return [[1, 2, 3]]

def test_init_builds_cpu_model_and_processor(monkeypatch):
    fake_model = _FakeLoadedModel()
    fake_processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000)
    )
    called = {"model": None, "processor": None}

    def fake_model_from_pretrained(*args, **kwargs):
        called["model"] = {"args": args, "kwargs": kwargs}
        return fake_model

    def fake_processor_from_pretrained(*args, **kwargs):
        called["processor"] = {"args": args, "kwargs": kwargs}
        return fake_processor

    monkeypatch.setattr(
        "src.asr.asr_model.AutoModelForSpeechSeq2Seq.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        "src.asr.asr_model.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )

    model = ASRModel(model_name="fake-model")

    assert model.torch_dtype == torch.float32
    assert fake_model.to_device == "cpu"
    assert called["model"]["args"] == ("fake-model",)
    assert called["processor"]["args"] == ("fake-model",)
    assert model.model is fake_model
    assert model.processor is fake_processor
    assert model.target_sample_rate == 16000


def test_transcribe_returns_clean_text_and_passes_kwargs():
    model = object.__new__(ASRModel)
    model.model = _FakeLoadedModel()
    model.target_sample_rate = 16000

    called = {"processor_audio": None, "processor_rate": None}

    def fake_load_wav(audio_input):
        assert audio_input == Path("sample.wav")
        return [0.1, 0.2], 16000

    def fake_processor(audio, sampling_rate, return_tensors):
        called["processor_audio"] = audio
        called["processor_rate"] = sampling_rate
        assert return_tensors == "pt"
        return {"input_features": "fake-features"}

    def fake_batch_decode(generated_ids, skip_special_tokens):
        assert generated_ids == [[1, 2, 3]]
        assert skip_special_tokens is True
        return [" test words "]

    model._load_wav = fake_load_wav
    model.processor = fake_processor
    model.processor.batch_decode = fake_batch_decode

    result = model.transcribe(Path("sample.wav"), generate_kwargs={"language": "en"})

    assert result == "test words"
    assert called["processor_audio"] == [0.1, 0.2]
    assert called["processor_rate"] == 16000
    assert model.model.generated == {
        "args": (),
        "kwargs": {"language": "en"},
    }


def test_transcribe_resamples_to_target_sample_rate():
    model = object.__new__(ASRModel)
    model.model = _FakeLoadedModel()
    model.target_sample_rate = 16000

    called = {"processor_rate": None}

    def fake_load_wav(_audio_input):
        return [0.1, 0.2, 0.3, 0.4], 24000

    def fake_processor(audio, sampling_rate, return_tensors):
        called["processor_rate"] = sampling_rate
        assert return_tensors == "pt"
        assert len(audio) > 0
        return {"input_features": "fake-features"}

    def fake_batch_decode(generated_ids, skip_special_tokens):
        assert generated_ids == [[1, 2, 3]]
        assert skip_special_tokens is True
        return [" ok "]

    model._load_wav = fake_load_wav
    model.processor = fake_processor
    model.processor.batch_decode = fake_batch_decode

    result = model.transcribe(Path("sample.wav"))

    assert result == "ok"
    assert called["processor_rate"] == 16000


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
