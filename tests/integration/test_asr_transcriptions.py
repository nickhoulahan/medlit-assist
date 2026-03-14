from pathlib import Path
import re
import wave

from src.asr.asr_model import ASRModel


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    no_punct = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(no_punct.split())


def test_asr_transcription_matches_fixture_text() -> None:

    model = ASRModel(model_name="openai/whisper-large-v3")
    
    # test normalized model response for each .wav file matches normalized text in corresponding .txt file
    for  file in (Path(__file__).parent / "testing_resources").glob("*.wav"):
        
        
        wav_path = file
        expected_text_path = file.with_name(file.stem.replace("question", "transcription") + ".txt")

        expected_text = expected_text_path.read_text(encoding="utf-8").strip()

        # Validate we can read the wav resource as part of the integration test.
        with wave.open(str(wav_path), "rb") as wav_file:
            assert wav_file.getnframes() > 0
            assert wav_file.getframerate() > 0

        text_response = model.generate_text_response(
            wav_path,
            generate_kwargs={"language": "en", "task": "transcribe"},
        )

        # text needs to be normalized due to common 
        assert _normalize_text(text_response) == _normalize_text(expected_text)
