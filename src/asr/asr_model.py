import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASRModel:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        torch_dtype: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype or torch.float32
        self._hf_token = os.getenv("HF_TOKEN")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=self._hf_token,
        ).to("cpu")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            token=self._hf_token,
        )
        self.asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.torch_dtype,
            device=-1,
        )

    def transcribe(
        self,
        audio_input: Path,
        generate_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        result = self.asr_pipeline(
            str(audio_input), generate_kwargs=dict(generate_kwargs or {})
        )

        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    def generate_text_response(
        self,
        audio_input: Path,
        generate_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.transcribe(audio_input=audio_input, generate_kwargs=generate_kwargs)
