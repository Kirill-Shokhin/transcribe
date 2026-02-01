import torch
import torchaudio
import gigaam
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps

from .config import settings
from .models import Segment, TranscribeResult


class ASRService:
    def __init__(self):
        self.asr_model = None
        self.vad_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_models(self):
        self.asr_model = gigaam.load_model(settings.model_type, device=self.device)
        self.vad_model = load_silero_vad()

    @property
    def is_ready(self) -> bool:
        return self.asr_model is not None and self.vad_model is not None

    def _merge_segments(self, timestamps: list[dict], sr: int) -> list[tuple[int, int]]:
        if not timestamps:
            return []

        chunks = []
        chunk_start = timestamps[0]['start']
        chunk_end = timestamps[0]['end']

        max_chunk = int(settings.max_chunk_duration * sr)
        max_gap = int(settings.max_gap_duration * sr)

        for ts in timestamps[1:]:
            gap = ts['start'] - chunk_end
            new_duration = ts['end'] - chunk_start

            if new_duration > max_chunk or gap > max_gap:
                chunks.append((chunk_start, chunk_end))
                chunk_start = ts['start']
            chunk_end = ts['end']

        chunks.append((chunk_start, chunk_end))
        return chunks

    def _transcribe_tensor(self, audio: torch.Tensor) -> str:
        """Transcribe tensor directly without saving to file."""
        wav = audio.to(self.asr_model._device).to(self.asr_model._dtype)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        length = torch.tensor([wav.shape[-1]], device=self.asr_model._device)
        encoded, encoded_len = self.asr_model.forward(wav, length)
        return self.asr_model.decoding.decode(self.asr_model.head, encoded, encoded_len)[0]

    def transcribe(self, audio_path: str | Path) -> TranscribeResult:
        audio_path = Path(audio_path)
        sr = settings.sample_rate

        wav, orig_sr = torchaudio.load(str(audio_path))
        wav = wav.squeeze(0)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)

        duration = wav.shape[0] / sr

        # Short audio — no segmentation
        if duration <= settings.short_audio_threshold:
            text = self._transcribe_tensor(wav)
            return TranscribeResult(
                text=text,
                segments=[Segment(start=0.0, end=duration, text=text)],
                duration=duration
            )

        # VAD segmentation
        timestamps = get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=sr,
            max_speech_duration_s=settings.max_chunk_duration,
            min_silence_duration_ms=settings.min_silence_duration_ms
        )

        if not timestamps:
            return TranscribeResult(text="", segments=[], duration=duration)

        chunks = self._merge_segments(timestamps, sr)

        segments = []
        texts = []

        for start, end in chunks:
            audio_chunk = wav[start:end]
            text = self._transcribe_tensor(audio_chunk)

            segments.append(Segment(
                start=start / sr,
                end=end / sr,
                text=text
            ))
            texts.append(text)

        return TranscribeResult(
            text=" ".join(texts),
            segments=segments,
            duration=duration
        )


asr_service = ASRService()
