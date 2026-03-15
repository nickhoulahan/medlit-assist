import io
from itertools import islice
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset


def main() -> None:
    """
    This script loads the ekacare/eka-medical-asr-evaluation-dataset from Hugging Face, extracts the first 5 samples
    It saves the audio as a .wav file with the corresponding transcript as a .txt file.
    """
    output_dir = Path(__file__).parent / "eval_resources"
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}

    dataset = load_dataset(
        "ekacare/eka-medical-asr-evaluation-dataset",
        split="test",
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))

    for index, sample in enumerate(islice(dataset, 5), start=1):
        base_file_name = f"sample_file_{index}"

        audio_info = sample["audio"]
        audio_bytes = audio_info.get("bytes")

        audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
        transcript = str(sample.get("text", "")).strip()

        wav_path = output_dir / f"{base_file_name}.wav"
        txt_path = output_dir / f"{base_file_name}.txt"

        sf.write(wav_path, audio_array, sampling_rate)
        txt_path.write_text(transcript + "\n", encoding="utf-8")

        print(f"[{index}] Saved {wav_path.name} and {txt_path.name}")


if __name__ == "__main__":
    main()
