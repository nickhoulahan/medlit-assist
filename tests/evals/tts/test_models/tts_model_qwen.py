from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


class TTSModel:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map="mps:0",
            dtype=torch.bfloat16,
        )

    def generate_speech(self, text: str, ref_audio: str, ref_text: str):
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return wavs, sr


if __name__ == "__main__":
    model = TTSModel()

    # Reference audio and text for voice cloning
    current_dir = Path(__file__).parent.__str__()
    ref_audio = f"{current_dir}/../test_assets/reference_audio.wav"
    ref_text = "What drug preventions are there for Alzheimer's disease?"
    model_input_text = open(
        Path(__file__).parent.parent / "test_assets" / "sample_text.txt"
    ).read()
    wavs, sr = model.generate_speech(
        text=model_input_text,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    output_file = Path(__file__).parent.parent / "test_output" / "tts_qwen.wav"
    sf.write(output_file, wavs[0], sr)
