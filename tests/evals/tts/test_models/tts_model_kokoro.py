from kokoro import KPipeline
from pathlib import Path
import soundfile as sf


class TTSModel:

    curr_dir = Path(__file__).parent.__str__()
    output_dir = Path(__file__).parent.parent / "test_output" / "tts_kokoro.wav"

    def __init__(self) -> None:
        self.pipeline = KPipeline(lang_code='a')
    
    def generate_speech(self, text: str):
        generator = self.pipeline(text, voice='af_heart')
        for gs, ps, audio in generator:
            sf.write(self.output_dir, audio, 24000)


if __name__ == "__main__":
    model = TTSModel()
    sample_text = open(
        Path(__file__).parent.parent / "test_assets" / "sample_text.txt"
    ).read()
    model.generate_speech(sample_text)
