import wave
from io import BytesIO
from pathlib import Path

import numpy as np
from kokoro import KPipeline


class TTSModel:
    """
    Text-to-speech model using Kokoro TTS.
    Args:
        voice (str): Voice identifier for Kokoro TTS.
        lang_code (str): Language code for Kokoro TTS.
        sample_rate (int): Sample rate for the generated audio.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize_speech_wav_bytes(self, text: str) -> tuple[bytes, int]:
        """Generate WAV bytes and sample rate from input text."""
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice):
            chunks.append(np.asarray(audio, dtype=np.float32))

        if not chunks:
            raise RuntimeError("Kokoro did not generate any audio")

        full_audio = np.concatenate(chunks)
        wav_bytes = self._pcm_f32_to_wav_bytes(full_audio, self.sample_rate)
        return wav_bytes, self.sample_rate

    def synthesize_to_wav_file(self, text: str, output_path: Path) -> Path:
        wav_bytes, _ = self.synthesize_speech_wav_bytes(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return output_path

    @staticmethod
    def _pcm_f32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())

        return buffer.getvalue()
