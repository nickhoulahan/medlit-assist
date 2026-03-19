import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
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


def test_init_raises_clear_error_for_unsupported_model(monkeypatch):
    def fake_model_from_pretrained(*_args, **_kwargs):
        raise ValueError("bad model")

    monkeypatch.setattr(
        "src.asr.asr_model.AutoModelForSpeechSeq2Seq.from_pretrained",
        fake_model_from_pretrained,
    )

    with pytest.raises(ValueError, match="Unsupported ASR model for this app"):
        ASRModel(model_name="not-a-seq2seq-model")


def test_resample_audio_handles_equal_rate_empty_and_short_target():
    same_rate = ASRModel._resample_audio(
        np.array([0.1, 0.2], dtype=np.float32),
        original_sample_rate=16000,
        target_sample_rate=16000,
    )
    assert np.allclose(same_rate, np.array([0.1, 0.2], dtype=np.float32))

    empty = ASRModel._resample_audio(
        np.array([], dtype=np.float32),
        original_sample_rate=24000,
        target_sample_rate=16000,
    )
    assert empty.size == 0

    short = ASRModel._resample_audio(
        np.array([0.5], dtype=np.float32),
        original_sample_rate=48000,
        target_sample_rate=16000,
    )
    assert short.shape[0] == 1
    assert np.allclose(short, np.array([0.5], dtype=np.float32))


def test_load_wav_mixes_to_mono_and_normalizes(tmp_path: Path):
    wav_path = tmp_path / "stereo.wav"
    pcm = np.array([32767, -32768, 1000, -1000], dtype=np.int16)

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm.tobytes())

    model = object.__new__(ASRModel)
    audio, sample_rate = model._load_wav(wav_path)

    assert sample_rate == 16000
    assert audio.shape[0] == 2
    assert audio.dtype == np.float32
    assert np.all(audio <= 1.0)
    assert np.all(audio >= -1.0)


def test_load_wav_rejects_non_16bit_sample_width(tmp_path: Path):
    wav_path = tmp_path / "bad-width.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x01")

    model = object.__new__(ASRModel)
    with pytest.raises(ValueError, match="Unsupported WAV sample width"):
        model._load_wav(wav_path)


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


def test_transcribe_uses_granite_processor_path():
    model = object.__new__(ASRModel)
    model.model = _FakeLoadedModel()
    model.target_sample_rate = 16000

    called = {"text": None, "audio": None, "return_tensors": None}

    def fake_load_wav(_audio_input):
        return np.array([0.25, -0.25], dtype=np.float32), 16000

    class GraniteSpeechProcessor:
        audio_token = "<|audio|>"

        def __call__(self, text=None, audio=None, return_tensors=None):
            called["text"] = text
            called["audio"] = audio
            called["return_tensors"] = return_tensors
            return {
                "input_features": torch.tensor([[1.0]]),
                "non_tensor": "ignore-me",
            }

        def batch_decode(self, generated_ids, skip_special_tokens=True):
            assert generated_ids == [[1, 2, 3]]
            assert skip_special_tokens is True
            return [" granite result "]

    model._load_wav = fake_load_wav
    model.processor = GraniteSpeechProcessor()

    result = model.transcribe(Path("sample.wav"))

    assert result == "granite result"
    assert called["text"] == "<|audio|>"
    assert isinstance(called["audio"], torch.Tensor)
    assert called["return_tensors"] == "pt"
    assert "input_features" in model.model.generated["kwargs"]
    assert "non_tensor" not in model.model.generated["kwargs"]


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
