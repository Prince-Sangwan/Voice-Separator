from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol
import tempfile

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.decomposition import FastICA
from transformers import pipeline


@dataclass(frozen=True)
class SeparationResult:
    sample_rate: int
    stream_paths: tuple[str, str]


class AudioSeparator(Protocol):
    def separate(self, audio_path: str | Path) -> SeparationResult:
        ...


class FastICASeparator:
    def __init__(self, target_sample_rate: int = 16000, n_components: int = 2) -> None:
        self.target_sample_rate = target_sample_rate
        self.n_components = n_components

    def separate(self, audio_path: str | Path) -> SeparationResult:
        waveform = self._load_audio(audio_path)
        observations = self._build_observation_matrix(waveform)
        separated_sources = self._run_ica(observations)
        stream_paths = self._write_streams(separated_sources)
        return SeparationResult(sample_rate=self.target_sample_rate, stream_paths=stream_paths)

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        waveform, _ = librosa.load(str(audio_path), sr=self.target_sample_rate, mono=False)
        return np.asarray(waveform, dtype=np.float32)

    def _build_observation_matrix(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 1:
            return self._build_mono_observation_matrix(waveform)
        channels = np.atleast_2d(waveform)
        if channels.shape[0] < self.n_components:
            return self._build_mono_observation_matrix(channels[0])
        return channels[: self.n_components].T.astype(np.float32)

    def _build_mono_observation_matrix(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size < self.n_components * 8:
            raise ValueError("The uploaded audio is too short to separate.")
        normalized = librosa.util.normalize(waveform)
        delayed = np.roll(normalized, 1)
        delayed[0] = 0.0
        emphasized = librosa.effects.preemphasis(normalized)
        observations = np.column_stack((normalized, 0.6 * delayed + 0.4 * emphasized))
        return observations.astype(np.float32)

    def _run_ica(self, observations: np.ndarray) -> np.ndarray:
        estimator = FastICA(
            n_components=self.n_components,
            whiten="unit-variance",
            max_iter=2000,
            tol=1e-4,
            random_state=0,
        )
        sources = estimator.fit_transform(observations)
        return sources.T.astype(np.float32)

    def _write_streams(self, sources: np.ndarray) -> tuple[str, str]:
        if sources.shape[0] != self.n_components:
            raise ValueError("The separator did not produce two output streams.")
        temp_dir = Path(tempfile.mkdtemp(prefix="voice_separator_"))
        output_paths: list[str] = []
        for index, source in enumerate(sources, start=1):
            output_path = temp_dir / f"speaker_{index}.wav"
            sf.write(str(output_path), self._normalize_audio(source), self.target_sample_rate)
            output_paths.append(str(output_path))
        return output_paths[0], output_paths[1]

    def _normalize_audio(self, source: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(source))) if source.size else 0.0
        if peak == 0.0:
            return source.astype(np.float32)
        return (source / peak).astype(np.float32)


class WhisperSmallTranscriber:
    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._lock = Lock()

    def transcribe(self, audio_path: str | Path) -> str:
        samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1, dtype=np.float32)
        audio_input = {
            "raw": np.ascontiguousarray(samples, dtype=np.float32),
            "sampling_rate": sample_rate,
        }
        result = self._get_pipeline()(audio_input, generate_kwargs={"task": "transcribe"})
        return result.get("text", "").strip()

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    device = 0 if torch.cuda.is_available() else -1
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    self._pipeline = pipeline(
                        task="automatic-speech-recognition",
                        model=self.model_name,
                        chunk_length_s=30,
                        device=device,
                        model_kwargs={"torch_dtype": torch_dtype},
                    )
        return self._pipeline


SEPARATOR: AudioSeparator = FastICASeparator()
TRANSCRIBER = WhisperSmallTranscriber()

CUSTOM_CSS = """
:root {
    --canvas: #07111f;
    --canvas-deep: #040a13;
    --surface: rgba(13, 29, 48, 0.78);
    --surface-strong: #0d1d31;
    --line: rgba(184, 219, 235, 0.15);
    --text: #f5fbff;
    --muted: #a9bdce;
    --mint: #8bf2c1;
    --mint-deep: #3bc995;
    --cyan: #7ad8ff;
    --amber: #ffc776;
    --rose: #ff9b87;
    --shadow: 0 30px 75px rgba(0, 0, 0, 0.34);
}

* {
    box-sizing: border-box;
}

html,
body,
.gradio-container {
    min-height: 100%;
    background: var(--canvas) !important;
}

body,
.gradio-container,
.gradio-container button,
.gradio-container input,
.gradio-container textarea {
    color: var(--text) !important;
    font-family: "Segoe UI Variable", "Aptos", "Trebuchet MS", sans-serif !important;
}

.gradio-container {
    position: relative;
    min-height: 100vh !important;
    max-width: none !important;
    overflow: hidden;
    padding: 34px clamp(18px, 4vw, 68px) 58px !important;
    background:
        radial-gradient(circle at 12% 8%, rgba(43, 129, 150, 0.24), transparent 32%),
        radial-gradient(circle at 88% 13%, rgba(60, 190, 138, 0.15), transparent 27%),
        linear-gradient(145deg, #081527 0%, #07111f 42%, #050b15 100%) !important;
}

.gradio-container::before,
.gradio-container::after {
    position: absolute;
    width: 34rem;
    height: 34rem;
    border-radius: 999px;
    content: "";
    filter: blur(5px);
    pointer-events: none;
}

.gradio-container::before {
    top: -17rem;
    right: -12rem;
    background: radial-gradient(circle, rgba(122, 216, 255, 0.16), transparent 66%);
    animation: ambient-drift 16s ease-in-out infinite alternate;
}

.gradio-container::after {
    bottom: -22rem;
    left: -14rem;
    background: radial-gradient(circle, rgba(139, 242, 193, 0.12), transparent 66%);
    animation: ambient-drift 20s ease-in-out infinite alternate-reverse;
}

#app-shell {
    position: relative;
    z-index: 1;
    width: min(1240px, 100%);
    margin: 0 auto;
    gap: 22px;
}

.hero-card {
    position: relative;
    overflow: hidden;
    min-height: 302px;
    padding: clamp(30px, 5vw, 54px);
    border: 1px solid rgba(164, 217, 240, 0.18);
    border-radius: 30px;
    background:
        linear-gradient(112deg, rgba(13, 33, 55, 0.96) 0%, rgba(11, 28, 45, 0.93) 58%, rgba(10, 48, 54, 0.82) 100%);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.hero-card::before {
    position: absolute;
    top: -38%;
    right: 7%;
    width: 27rem;
    height: 27rem;
    border: 1px solid rgba(139, 242, 193, 0.18);
    border-radius: 50%;
    content: "";
}

.hero-card::after {
    position: absolute;
    right: 12%;
    bottom: -44%;
    width: 30rem;
    height: 30rem;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(89, 211, 191, 0.19), transparent 64%);
    content: "";
}

.hero-grid {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(208px, 0.38fr);
    align-items: center;
    gap: 34px;
}

.hero-kicker,
.step-label,
.stream-eyebrow,
.transcript-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: var(--mint);
    font-size: 0.72rem;
    font-weight: 760;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.hero-kicker::before {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--mint);
    box-shadow: 0 0 0 5px rgba(139, 242, 193, 0.12), 0 0 18px rgba(139, 242, 193, 0.85);
    content: "";
}

.hero-card h1 {
    max-width: 710px;
    margin: 17px 0 15px;
    color: var(--text);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(2.65rem, 5.6vw, 5rem);
    font-weight: 500;
    letter-spacing: -0.065em;
    line-height: 0.98;
}

.hero-card h1 em {
    color: var(--mint);
    font-style: italic;
}

.hero-card p {
    max-width: 620px;
    margin: 0;
    color: var(--muted);
    font-size: 1.03rem;
    line-height: 1.7;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 25px;
}

.hero-badge,
.signal-chip,
.stream-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(176, 219, 235, 0.17);
    border-radius: 999px;
    background: rgba(5, 17, 30, 0.38);
    color: #d8e9f4;
    font-size: 0.78rem;
    font-weight: 650;
    letter-spacing: 0.01em;
}

.hero-badge {
    padding: 9px 12px;
}

.hero-badge span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 10px rgba(122, 216, 255, 0.8);
}

.hero-art {
    position: relative;
    display: grid;
    min-height: 194px;
    place-items: center;
}

.hero-art-halo {
    position: absolute;
    width: 178px;
    height: 178px;
    border: 1px solid rgba(139, 242, 193, 0.26);
    border-radius: 48% 52% 45% 55%;
    box-shadow: inset 0 0 54px rgba(122, 216, 255, 0.08), 0 0 48px rgba(139, 242, 193, 0.12);
    animation: halo-spin 18s linear infinite;
}

.hero-art svg {
    position: relative;
    z-index: 1;
    width: 180px;
    filter: drop-shadow(0 12px 30px rgba(122, 216, 255, 0.2));
}

.hero-art-orbit {
    position: absolute;
    top: 10px;
    right: 12%;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--amber);
    box-shadow: 0 0 0 7px rgba(255, 199, 118, 0.11), 0 0 22px rgba(255, 199, 118, 0.78);
    animation: soft-pulse 2.8s ease-in-out infinite;
}

#upload-panel {
    overflow: hidden;
    padding: clamp(22px, 3.5vw, 34px) !important;
    border: 1px solid var(--line) !important;
    border-radius: 24px !important;
    background: linear-gradient(145deg, rgba(16, 36, 58, 0.82), rgba(11, 25, 42, 0.76)) !important;
    box-shadow: 0 18px 46px rgba(0, 0, 0, 0.17), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

#upload-panel .form,
#upload-panel .wrap {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.panel-intro,
.results-heading,
.privacy-note {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.panel-intro {
    margin-bottom: 22px;
}

.panel-intro h2,
.results-heading h2 {
    margin: 7px 0 6px;
    color: var(--text);
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(1.65rem, 2.6vw, 2.2rem);
    font-weight: 500;
    letter-spacing: -0.04em;
}

.panel-intro p,
.results-heading p {
    margin: 0;
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.55;
}

.signal-chip {
    flex: 0 0 auto;
    padding: 10px 14px;
    color: #eaf9ff;
}

.signal-chip svg {
    width: 18px;
    height: 18px;
    stroke: var(--cyan);
}

.workflow-steps {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 0 0 19px;
}

.workflow-step {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
    padding: 10px 12px;
    border: 1px solid rgba(184, 219, 235, 0.1);
    border-radius: 13px;
    background: rgba(5, 15, 28, 0.36);
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 600;
}

.workflow-step strong {
    display: grid;
    flex: 0 0 auto;
    width: 23px;
    height: 23px;
    place-items: center;
    border-radius: 8px;
    background: rgba(139, 242, 193, 0.12);
    color: var(--mint);
    font-size: 0.72rem;
}

#audio-upload {
    min-height: 164px;
    border: 1px dashed rgba(139, 242, 193, 0.5) !important;
    border-radius: 18px !important;
    background: linear-gradient(135deg, rgba(10, 30, 43, 0.9), rgba(8, 21, 37, 0.72)) !important;
    transition: border-color 180ms ease, transform 180ms ease, background 180ms ease;
}

#audio-upload:hover {
    border-color: var(--mint) !important;
    background: linear-gradient(135deg, rgba(13, 42, 54, 0.94), rgba(9, 27, 43, 0.8)) !important;
    transform: translateY(-1px);
}

#audio-upload .wrap,
#audio-upload .file-preview,
#audio-upload [data-testid="file-preview"],
#audio-upload [data-testid="file-upload-dropzone"] {
    background: transparent !important;
}

#audio-upload button {
    border: 1px solid rgba(139, 242, 193, 0.45) !important;
    border-radius: 10px !important;
    background: rgba(139, 242, 193, 0.11) !important;
    color: var(--mint) !important;
    font-weight: 700 !important;
}

.processing-note {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 18px 0 12px;
    color: var(--muted);
    font-size: 0.83rem;
}

.processing-note span {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--mint);
    box-shadow: 0 0 13px rgba(139, 242, 193, 0.9);
}

#process-button,
#process-button button {
    width: 100% !important;
    min-height: 58px;
    border: 0 !important;
    border-radius: 15px !important;
    background: linear-gradient(100deg, #8df2c1 0%, #78e3dc 52%, #7cc4f5 100%) !important;
    box-shadow: 0 15px 28px rgba(76, 210, 176, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.64) !important;
    color: #062231 !important;
    font-size: 0.98rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em;
    transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease !important;
}

#process-button:hover,
#process-button button:hover {
    box-shadow: 0 19px 34px rgba(76, 210, 176, 0.31), inset 0 1px 0 rgba(255, 255, 255, 0.78) !important;
    filter: saturate(1.08) brightness(1.03);
    transform: translateY(-2px);
}

#process-button:active,
#process-button button:active {
    transform: translateY(0);
}

.results-heading {
    margin: 7px 3px 1px;
}

.results-heading p {
    max-width: 420px;
    text-align: right;
}

#results-grid {
    align-items: stretch;
    gap: 22px !important;
}

#speaker-one-card,
#speaker-two-card {
    min-width: 0;
    padding: clamp(20px, 3vw, 28px) !important;
    border: 1px solid var(--line);
    border-radius: 24px;
    background: linear-gradient(155deg, rgba(15, 37, 58, 0.9), rgba(9, 22, 38, 0.82));
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.17), inset 0 1px 0 rgba(255, 255, 255, 0.055);
}

#speaker-one-card {
    border-top: 2px solid rgba(139, 242, 193, 0.8);
}

#speaker-two-card {
    border-top: 2px solid rgba(255, 199, 118, 0.86);
}

.stream-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 21px;
}

.stream-identity {
    display: flex;
    align-items: center;
    gap: 12px;
}

.stream-index {
    display: grid;
    width: 40px;
    height: 40px;
    place-items: center;
    border-radius: 13px;
    background: rgba(139, 242, 193, 0.13);
    color: var(--mint);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}

#speaker-two-card .stream-index {
    background: rgba(255, 199, 118, 0.14);
    color: var(--amber);
}

.stream-header h3 {
    margin: 3px 0 0;
    color: var(--text);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.43rem;
    font-weight: 500;
    letter-spacing: -0.035em;
}

.stream-status {
    padding: 7px 9px;
    color: var(--mint);
    font-size: 0.68rem;
    white-space: nowrap;
}

#speaker-two-card .stream-status {
    color: var(--amber);
}

.stream-status i {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 10px currentColor;
}

#speaker-one-audio,
#speaker-two-audio {
    overflow: hidden;
    margin-bottom: 18px;
    border: 1px solid rgba(184, 219, 235, 0.12) !important;
    border-radius: 16px !important;
    background: rgba(4, 14, 27, 0.55) !important;
}

#speaker-one-audio .wrap,
#speaker-two-audio .wrap,
#speaker-one-audio .audio-container,
#speaker-two-audio .audio-container {
    background: transparent !important;
}

.transcript-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 3px 0 9px;
}

.transcript-heading span:last-child {
    color: #7d9aac;
    font-size: 0.73rem;
}

#speaker-one-text textarea,
#speaker-two-text textarea {
    min-height: 172px !important;
    border: 1px solid rgba(184, 219, 235, 0.12) !important;
    border-radius: 16px !important;
    background: rgba(4, 14, 27, 0.56) !important;
    color: #eaf5fa !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

#speaker-one-text textarea::placeholder,
#speaker-two-text textarea::placeholder {
    color: #7893a4 !important;
}

.privacy-note {
    padding: 2px 4px 0;
    color: #7893a4;
    font-size: 0.8rem;
}

.privacy-note strong {
    color: var(--mint);
    font-weight: 750;
}

footer {
    position: relative;
    z-index: 1;
    margin-top: 12px !important;
    color: #7590a1 !important;
}

footer a {
    color: #a7c8d8 !important;
}

@keyframes ambient-drift {
    from {
        transform: translate3d(0, 0, 0) scale(1);
    }
    to {
        transform: translate3d(-28px, 21px, 0) scale(1.1);
    }
}

@keyframes halo-spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes soft-pulse {
    0%,
    100% {
        transform: scale(0.9);
        opacity: 0.74;
    }
    50% {
        transform: scale(1.12);
        opacity: 1;
    }
}

@media (max-width: 760px) {
    html,
    body,
    .gradio-container {
        width: 100% !important;
        min-width: 0 !important;
        overflow-x: hidden !important;
    }

    .gradio-container {
        padding: 18px 14px 38px !important;
    }

    #app-shell,
    #app-shell > *,
    #upload-panel,
    #results-grid,
    #speaker-one-card,
    #speaker-two-card {
        min-width: 0 !important;
        max-width: 100% !important;
    }

    #app-shell {
        width: 100% !important;
    }

    .hero-card {
        min-height: 0;
        padding: 30px 24px;
        border-radius: 23px;
    }

    .hero-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }

    .hero-art {
        display: none;
    }

    .hero-card h1 {
        font-size: clamp(2.45rem, 14vw, 3.55rem);
    }

    .hero-badge {
        max-width: 100%;
    }

    .panel-intro,
    .results-heading,
    .privacy-note {
        align-items: flex-start;
        flex-direction: column;
        gap: 12px;
    }

    .results-heading p {
        text-align: left;
    }

    .signal-chip {
        width: 100%;
    }

    .workflow-steps {
        grid-template-columns: 1fr;
    }

    #results-grid {
        flex-direction: column !important;
        gap: 16px !important;
    }

    #speaker-one-card,
    #speaker-two-card {
        width: 100% !important;
    }

    #process-button,
    #process-button button {
        min-width: 0 !important;
    }

    .stream-status {
        display: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
"""


def process_audio(audio_file: str) -> tuple[str, str, str, str]:
    if not audio_file:
        raise gr.Error("Upload an audio file to begin processing.")
    try:
        separation_result = SEPARATOR.separate(audio_file)
        speaker_one_path, speaker_two_path = separation_result.stream_paths
        speaker_one_text = TRANSCRIBER.transcribe(speaker_one_path)
        speaker_two_text = TRANSCRIBER.transcribe(speaker_two_path)
        return speaker_one_path, speaker_one_text, speaker_two_path, speaker_two_text
    except Exception as exc:
        raise gr.Error(f"Processing failed: {exc}") from exc


with gr.Blocks(title="Overlapping Voice Separator and Transcriber", fill_width=True) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <section class="hero-card">
                <div class="hero-grid">
                    <div>
                        <div class="hero-kicker">Voice intelligence studio</div>
                        <h1>Hear the conversation,<br><em>not the overlap.</em></h1>
                        <p>Separate a layered recording into two listening streams, then turn each voice into a searchable transcript.</p>
                        <div class="hero-badges">
                            <div class="hero-badge"><span></span>Two output streams</div>
                            <div class="hero-badge"><span></span>Whisper transcription</div>
                            <div class="hero-badge"><span></span>Local processing</div>
                        </div>
                    </div>
                    <div class="hero-art" aria-hidden="true">
                        <div class="hero-art-halo"></div>
                        <svg viewBox="0 0 220 160" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M8 80H30L43 36L58 126L75 56L91 106L108 20L124 140L140 72L152 102L166 47L180 116L193 80H212" stroke="#8BF2C1" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                            <path d="M8 92H32L46 76L60 105L76 74L92 102L108 56L124 112L141 85L154 96L168 70L181 101L194 88H212" stroke="#7AD8FF" stroke-opacity="0.78" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                        <div class="hero-art-orbit"></div>
                    </div>
                </div>
            </section>
            """
        )
        with gr.Column(elem_id="upload-panel"):
            gr.HTML(
                """
                <div class="panel-intro">
                    <div>
                        <div class="step-label">Step 01</div>
                        <h2>Bring in your audio</h2>
                        <p>Upload one conversation recording. We will prepare it at 16 kHz before analysis.</p>
                    </div>
                    <div class="signal-chip">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 12H7L9.5 6L13 18L15.5 9L17 12H20" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
                        Ready for signal processing
                    </div>
                </div>
                <div class="workflow-steps">
                    <div class="workflow-step"><strong>01</strong> Upload source audio</div>
                    <div class="workflow-step"><strong>02</strong> Isolate two streams</div>
                    <div class="workflow-step"><strong>03</strong> Generate transcripts</div>
                </div>
                """
            )
            input_audio = gr.File(
                label="Conversation audio",
                file_types=["audio"],
                type="filepath",
                height=164,
                show_label=False,
                elem_id="audio-upload",
            )
            gr.HTML(
                """
                <div class="processing-note"><span></span>Your audio stays on this machine while it is processed.</div>
                """
            )
            process_button = gr.Button(
                "Separate and transcribe",
                variant="primary",
                size="lg",
                elem_id="process-button",
            )
        gr.HTML(
            """
            <div class="results-heading">
                <div>
                    <div class="step-label">Step 02</div>
                    <h2>Review the separated voices</h2>
                </div>
                <p>Listen to each stream and use the generated text to compare the results.</p>
            </div>
            """
        )
        with gr.Row(elem_id="results-grid", equal_height=True):
            with gr.Column(elem_id="speaker-one-card"):
                gr.HTML(
                    """
                    <div class="stream-header">
                        <div class="stream-identity">
                            <div class="stream-index">01</div>
                            <div>
                                <div class="stream-eyebrow">Output channel</div>
                                <h3>Voice one</h3>
                            </div>
                        </div>
                        <div class="stream-status"><i></i>Ready to review</div>
                    </div>
                    """
                )
                speaker_one_audio = gr.Audio(
                    label="Voice one audio",
                    type="filepath",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="speaker-one-audio",
                )
                gr.HTML(
                    """
                    <div class="transcript-heading">
                        <div class="transcript-eyebrow">Transcript</div>
                        <span>Whisper Small</span>
                    </div>
                    """
                )
                speaker_one_text = gr.Textbox(
                    label="Voice one transcript",
                    lines=8,
                    max_lines=14,
                    placeholder="The transcript for voice one will appear here.",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="speaker-one-text",
                )
            with gr.Column(elem_id="speaker-two-card"):
                gr.HTML(
                    """
                    <div class="stream-header">
                        <div class="stream-identity">
                            <div class="stream-index">02</div>
                            <div>
                                <div class="stream-eyebrow">Output channel</div>
                                <h3>Voice two</h3>
                            </div>
                        </div>
                        <div class="stream-status"><i></i>Ready to review</div>
                    </div>
                    """
                )
                speaker_two_audio = gr.Audio(
                    label="Voice two audio",
                    type="filepath",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="speaker-two-audio",
                )
                gr.HTML(
                    """
                    <div class="transcript-heading">
                        <div class="transcript-eyebrow">Transcript</div>
                        <span>Whisper Small</span>
                    </div>
                    """
                )
                speaker_two_text = gr.Textbox(
                    label="Voice two transcript",
                    lines=8,
                    max_lines=14,
                    placeholder="The transcript for voice two will appear here.",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_id="speaker-two-text",
                )
        gr.HTML(
            """
            <div class="privacy-note">
                <span><strong>Private by design.</strong> Separation and transcription run locally on your machine.</span>
                <span>FastICA plus Whisper Small</span>
            </div>
            """
        )

    process_button.click(
        fn=process_audio,
        inputs=input_audio,
        outputs=[speaker_one_audio, speaker_one_text, speaker_two_audio, speaker_two_text],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=gr.themes.Base(), css=CUSTOM_CSS)
