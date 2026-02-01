import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torchaudio
from transformers import AutoModel
from silero_vad import load_silero_vad, get_speech_timestamps


def merge_segments(timestamps: list, sr: int, max_chunk: float = 25.0, max_gap: float = 1.5):
    """Склеивает мелкие сегменты в куски до max_chunk секунд."""
    if not timestamps:
        return []

    chunks = []
    chunk_start = timestamps[0]['start']
    chunk_end = timestamps[0]['end']

    for ts in timestamps[1:]:
        gap = (ts['start'] - chunk_end) / sr
        new_duration = (ts['end'] - chunk_start) / sr

        # Новый чанк если: превысит лимит или большая пауза
        if new_duration > max_chunk or gap > max_gap:
            chunks.append((chunk_start, chunk_end))
            chunk_start = ts['start']
        chunk_end = ts['end']

    chunks.append((chunk_start, chunk_end))
    return chunks


def transcribe_long(audio_path: str, asr_model, vad_model, sr: int = 16000):
    """Транскрибирует аудио с VAD-сегментацией."""

    wav, orig_sr = torchaudio.load(audio_path)
    wav = wav.squeeze(0)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)

    duration = wav.shape[0] / sr
    print(f"Duration: {duration:.1f}s")

    # Короткое аудио - без сегментации
    if duration <= 30:
        return asr_model.transcribe(audio_path)

    # VAD
    timestamps = get_speech_timestamps(wav, vad_model,
        sampling_rate=sr,
        max_speech_duration_s=25.0,
        min_silence_duration_ms=300
    )

    if not timestamps:
        return ""

    chunks = merge_segments(timestamps, sr)
    print(f"Segments: {len(chunks)}")

    results = []
    for start, end in chunks:
        audio_chunk = wav[start:end]
        tmp_path = "_temp_chunk.wav"
        torchaudio.save(tmp_path, audio_chunk.unsqueeze(0), sr)

        text = asr_model.transcribe(tmp_path)
        results.append(text)
        print(f"  [{start/sr:.1f}s-{end/sr:.1f}s]: {text}")

    return " ".join(results)


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "test.wav"

    print("Loading models...")
    asr_model = AutoModel.from_pretrained(
        "ai-sage/GigaAM-v3",
        revision="e2e_rnnt",
        trust_remote_code=True,
    )
    vad_model = load_silero_vad()

    print(f"Transcribing: {audio_path}")
    result = transcribe_long(audio_path, asr_model, vad_model)

    print(f"\n=== RESULT ===\n{result}")


if __name__ == "__main__":
    main()
