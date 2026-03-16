import wave
from pathlib import Path

from piper import PiperVoice


class TTSModel:

    curr_dir = Path(__file__).parent.__str__()
    output_dir = Path(__file__).parent.parent / "test_output" / "tts_piper.wav"

    def generate_speech(self, text: str):

        voice = PiperVoice.load(
            f"{self.curr_dir}/../test_assets/en_US-kathleen-low.onnx"
        )
        with wave.open(self.output_dir.__str__(), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)


if __name__ == "__main__":
    model = TTSModel()
    sample_text = open(
        Path(__file__).parent.parent / "test_assets" / "sample_text.txt"
    ).read()
    model.generate_speech(sample_text)
