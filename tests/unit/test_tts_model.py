from pathlib import Path
from io import BytesIO
import wave

import pytest

from src.tts.tts_model import TTSModel


def test_synthesize_speech_wav_bytes_rejects_empty_text():
    model = TTSModel()

    with pytest.raises(ValueError, match="non-empty"):
        model.synthesize_speech_wav_bytes("   ")


def test_synthesize_speech_wav_bytes_uses_piper_voice(monkeypatch):
    model = TTSModel()

    class _FakeVoice:
        def synthesize_wav(self, text, wav_file):
            assert text == "hello"
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 8)

    monkeypatch.setattr(model, "voice", _FakeVoice())

    wav_bytes, sr = model.synthesize_speech_wav_bytes("hello")

    assert wav_bytes[:4] == b"RIFF"
    assert sr == 22050
    with wave.open(BytesIO(wav_bytes), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 22050

def test_synthesize_to_wav_file_writes_file(tmp_path: Path, monkeypatch):
    model = TTSModel()

    monkeypatch.setattr(
        model,
        "synthesize_speech_wav_bytes",
        lambda text: (b"RIFFbytes", 22050),
    )

    output = tmp_path / "sample.wav"
    result = model.synthesize_to_wav_file("test", output)

    assert result == output
    assert output.exists()
    assert output.read_bytes() == b"RIFFbytes"
