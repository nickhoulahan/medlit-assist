import os
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from scipy import signal
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class ASRModel:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        torch_dtype: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype or torch.float32
        self._hf_token = os.getenv("HF_TOKEN")

        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=self._hf_token,
            ).to("cpu")
        except ValueError as exc:
            raise ValueError(
                f"Unsupported ASR model for this app: '{self.model_name}'. "
                "This ASR wrapper expects a Hugging Face Transformers SpeechSeq2Seq "
                "model (for example openai/whisper-large-v3)."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self._hf_token,
        )
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        self.target_sample_rate = int(
            getattr(feature_extractor, "sampling_rate", 16000)
        )

    @staticmethod
    def _resample_audio(
        audio: np.ndarray,
        original_sample_rate: int,
        target_sample_rate: int,
    ) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if original_sample_rate == target_sample_rate or audio.size == 0:
            return audio

        resampled = signal.resample_poly(
            audio, target_sample_rate, original_sample_rate
        )

        return resampled.astype(np.float32)

    def _load_wav(self, audio_input: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(audio_input), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw_frames = wav_file.readframes(wav_file.getnframes())

        # check for supported sample width (16-bit PCM)
        if sample_width == 2:
            audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        else:
            raise ValueError(
                "Unsupported WAV sample width. Expected 16-bit PCM WAV "
                f"(sample width 2), got {sample_width}."
            )

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio, sample_rate

    def transcribe(
        self,
        audio_input: Path,
        generate_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Transcribes the given audio input using the ASR model.
        """
        audio, sample_rate = self._load_wav(audio_input)
        target_sample_rate = getattr(self, "target_sample_rate", 16000)
        if sample_rate != target_sample_rate:
            audio = self._resample_audio(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate

        # granite requires a special audio token and different processor kwargs
        # this is added to accommodate evaluation of granite ASR model
        processor_name = self.processor.__class__.__name__
        if processor_name == "GraniteSpeechProcessor":
            audio_token = getattr(self.processor, "audio_token", "<|audio|>")
            inputs = self.processor(
                text=audio_token,
                audio=torch.tensor(audio),
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )

        model_inputs = {
            key: value
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **dict(generate_kwargs or {}),
            )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    def generate_text_response(
        self,
        audio_input: Path,
        generate_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.transcribe(audio_input=audio_input, generate_kwargs=generate_kwargs)
