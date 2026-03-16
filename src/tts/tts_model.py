import wave
from io import BytesIO
from pathlib import Path

from piper import PiperVoice


class TTSModel:
    def __init__(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.model_path = repo_root / "src/tts/assets/en_US-kathleen-low.onnx"
        self.voice: PiperVoice = PiperVoice.load(str(self.model_path))

    def synthesize_speech_wav_bytes(self, text: str) -> tuple[bytes, int]:
        """Generate WAV bytes and sample rate from input text using Piper."""
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        buffer = BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        wav_bytes = buffer.getvalue()
        with wave.open(BytesIO(wav_bytes), "rb") as wav_reader:
            sr = int(wav_reader.getframerate())

        return wav_bytes, sr

    def synthesize_to_wav_file(self, text: str, output_path: Path) -> Path:
        wav_bytes, _ = self.synthesize_speech_wav_bytes(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return output_path
