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

    def _transcribe_chunk(self, audio: torch.Tensor, sr: int) -> str:
        torchaudio.save(str(settings.temp_chunk_path), audio.unsqueeze(0), sr)
        text = self.asr_model.transcribe(str(settings.temp_chunk_path))
        settings.temp_chunk_path.unlink(missing_ok=True)
        return text

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
            text = self.asr_model.transcribe(str(audio_path))
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
            text = self._transcribe_chunk(audio_chunk, sr)

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
