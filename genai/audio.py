"""Local speech-to-text with Whisper, via the transformers pipeline.

Audio is the third modality. Whisper turns a waveform into a log-Mel
spectrogram (``log_mel``) and reads that picture of sound the way a vision model
reads an image, emitting text (``transcribe``) or text with word-level timing
(``transcribe_timed``). We decode WAV files with the standard library so there
is no ffmpeg dependency to install.
"""
import functools
import wave

import numpy as np

DEFAULT_ASR_MODEL = "openai/whisper-base"


def read_wav(path):
    """Read a mono 16-bit PCM WAV into (float32 samples in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, sr


@functools.lru_cache(maxsize=1)
def _asr(model: str = DEFAULT_ASR_MODEL):
    """Load and cache the speech-recognition pipeline (downloaded once)."""
    from transformers import pipeline
    from transformers.utils import logging
    logging.set_verbosity_error()
    return pipeline("automatic-speech-recognition", model=model)


def transcribe(wav_path, model: str = DEFAULT_ASR_MODEL) -> str:
    """Transcribe a 16 kHz mono WAV file to text using a local Whisper model."""
    audio, sr = read_wav(wav_path)
    result = _asr(model)({"array": audio, "sampling_rate": sr},
                         generate_kwargs={"language": "en", "task": "transcribe"})
    return result["text"].strip()


def asr_size(model: str = DEFAULT_ASR_MODEL):
    """Return how many parameters the default Whisper model carries, in millions."""
    weights = _asr(model).model.parameters()
    return sum(w.numel() for w in weights) / 1e6


def log_mel(wav_path, model: str = DEFAULT_ASR_MODEL):
    """Return the log-Mel spectrogram Whisper actually ingests: an 80-row picture
    of which pitches are loud over time, cropped to the clip's true length (the
    feature extractor pads every clip out to its 30-second window)."""
    audio, sr = read_wav(wav_path)
    fe = _asr(model).feature_extractor
    mel = fe(audio, sampling_rate=sr, return_tensors="np")["input_features"][0]
    keep = round(len(audio) / sr / fe.chunk_length * mel.shape[1])
    return mel[:, :keep]


def transcribe_timed(wav_path, model: str = DEFAULT_ASR_MODEL):
    """Transcribe with word-level timing: a list of (start, end, word) tuples,
    the raw material behind every set of automatic captions."""
    audio, sr = read_wav(wav_path)
    result = _asr(model)({"array": audio, "sampling_rate": sr},
                         return_timestamps="word",
                         generate_kwargs={"language": "en", "task": "transcribe"})
    return [(c["timestamp"][0], c["timestamp"][1], c["text"].strip())
            for c in result["chunks"]]


def show_word_timings(words, n: int = 6) -> None:
    """Print the first ``n`` word-level timestamps, then the total word count."""
    for start, end, word in words[:n]:
        print(f"{start:4.1f}-{end:4.1f}s  {word}")
    print(f"... {len(words)} words total")
