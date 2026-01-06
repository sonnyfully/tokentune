from future import annotations

import torch

def slice_audio(waveform: torch.Tensor,
                sample_rate: int,
                clip_seconds: float,
                hop_seconds: float,
                ) -> list[torch.Tensor]:
    
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError(f"Waveform must be a 2D tensor with shape (1, N), got {waveform.shape}")
    
    clip_samples = int(round(clip_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))

    if clip_samples <= 0 or hop_samples <= 0:
        raise ValueError(f"Clip and hop samples must be positive, got {clip_samples} and {hop_samples}")
    
    T = waveform.size(1)
    clips = []
    
    start = 0
    while start + clip_samples <= T:
        clip = waveform[:, start:start+clip_samples]
        clips.append(clip)
        start += hop_samples
    
    return clips