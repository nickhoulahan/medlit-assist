import io
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline


class TTSModel:
    def __init__(self, model_id: str = "facebook/mms-tts-eng") -> None:
        self.model_id = model_id
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipe: Any | None = None

    def get_tts_pipe(self):
        if self._pipe is None:
            self._pipe = pipeline(
                task="text-to-speech",
                model=self.model_id,
                device=self.device,
            )
        return self._pipe

    def synthesize_speech_wav_bytes(self, text: str) -> tuple[bytes, int]:

        tts = self.get_tts_pipe()
        out = tts(text)

        audio = np.asarray(out["audio"]).squeeze()
        sr = int(out["sampling_rate"])

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        dur = float(audio.size / sr) if sr and audio.size else 0.0

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), sr


if __name__ == "__main__":
    tts_model = TTSModel()
    output = Path(__file__).parent.parent / "test_output" / "tts_facebook.wav"
    sample_text = open(
        Path(__file__).parent.parent / "test_assets" / "sample_text.txt"
    ).read()
    wav_bytes, sr = tts_model.synthesize_speech_wav_bytes(sample_text)
    with open(output, "wb") as f:
        f.write(wav_bytes)
