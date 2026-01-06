from future import annotations

import soundfile as sf
import torch
import torchaudio
from pathlib import Path

def load_audio(path: Path | str) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)
    return waveform, sample_rate

def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() != 2:
        raise ValueError(f"Waveform must be a 2D tensor, got {waveform.dim()}D")
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)

def resample(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if sample_rate == target_sample_rate:
        return waveform
        
    return torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    peak = waveform.abs().max().item()
    if peak < eps:
        return waveform
    return waveform * (target_peak / peak)

def rms(waveform: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.sqrt(torch.mean(waveform**2) + eps),item())

def save_wav(wav: torch.Tensor, sample_rate: int, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    wav_np = wav.detach().cpu().numpy()
    sf.write(str(path), wav_np, sample_rate, format="WAV")