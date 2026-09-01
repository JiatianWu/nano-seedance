"""Mel spectrogram front-end.

The audio branch of the DiT operates on log-mel spectrograms rather than raw
waveforms: mel frames are ~100x cheaper than samples and still carry the
onset structure that audio-visual sync depends on. Griffin-Lim gives us a
vocoder-free path back to a listenable waveform.

torchaudio supplies the filterbank and Griffin-Lim when installed; equivalent
pure-torch implementations are kept as a fallback so the pipeline has no hard
dependency on it. Normalisation, framing and tensor layout live here either
way, so both paths produce identical shapes and value ranges.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:  # optional
    import torchaudio.functional as taF

    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover - exercised only when torchaudio is absent
    taF = None
    _HAS_TORCHAUDIO = False

# Fixed log-domain range so mel tensors land in [-1, 1] like the video tensors.
MEL_LOG_MIN = -11.5
MEL_LOG_MAX = 3.0


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Triangular mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    f_max = f_max if f_max is not None else sample_rate / 2
    n_freqs = n_fft // 2 + 1

    if _HAS_TORCHAUDIO:
        fb = taF.melscale_fbanks(
            n_freqs=n_freqs, f_min=f_min, f_max=f_max, n_mels=n_mels,
            sample_rate=sample_rate, norm="slaney", mel_scale="htk",
        )  # (n_freqs, n_mels)
        return fb.transpose(0, 1).to(device=device)

    freqs = torch.linspace(0, sample_rate / 2, n_freqs, device=device)

    mel_pts = torch.linspace(
        float(hz_to_mel(torch.tensor(f_min))),
        float(hz_to_mel(torch.tensor(f_max))),
        n_mels + 2,
        device=device,
    )
    hz_pts = mel_to_hz(mel_pts)

    lower = hz_pts[:-2].unsqueeze(1)
    center = hz_pts[1:-1].unsqueeze(1)
    upper = hz_pts[2:].unsqueeze(1)
    f = freqs.unsqueeze(0)

    up_slope = (f - lower) / torch.clamp(center - lower, min=1e-8)
    down_slope = (upper - f) / torch.clamp(upper - center, min=1e-8)
    fb = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)

    # Slaney-style area normalisation keeps loudness comparable across bands.
    enorm = 2.0 / torch.clamp(upper.squeeze(1) - lower.squeeze(1), min=1e-8)
    return fb * enorm.unsqueeze(1)


class MelSpectrogram:
    """Waveform <-> normalised log-mel, with a Griffin-Lim inverse."""

    def __init__(
        self,
        sample_rate: int = 8000,
        n_fft: int = 256,
        hop_length: int = 128,
        n_mels: int = 32,
        n_frames: int = 128,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_frames = n_frames
        self._fb: Optional[torch.Tensor] = None
        self._window: Optional[torch.Tensor] = None

    def _cached(self, device: torch.device):
        if self._fb is None or self._fb.device != device:
            self._fb = mel_filterbank(self.sample_rate, self.n_fft, self.n_mels, device=device)
            self._window = torch.hann_window(self.n_fft, device=device)
        return self._fb, self._window

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(..., samples) -> (..., n_mels, n_frames) normalised to [-1, 1]."""
        squeeze = waveform.dim() == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        fb, window = self._cached(waveform.device)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.abs() ** 2
        mel = torch.matmul(fb, power)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = (mel - MEL_LOG_MIN) / (MEL_LOG_MAX - MEL_LOG_MIN) * 2.0 - 1.0
        mel = mel.clamp(-1, 1)
        mel = _fit_frames(mel, self.n_frames)
        return mel.squeeze(0) if squeeze else mel

    def inverse(self, mel: torch.Tensor, num_samples: Optional[int] = None,
                n_iter: int = 32) -> torch.Tensor:
        """Normalised log-mel -> waveform via pseudo-inverse + Griffin-Lim."""
        squeeze = mel.dim() == 2
        if squeeze:
            mel = mel.unsqueeze(0)
        fb, window = self._cached(mel.device)

        log_mel = (mel.clamp(-1, 1) + 1.0) / 2.0 * (MEL_LOG_MAX - MEL_LOG_MIN) + MEL_LOG_MIN
        mel_power = torch.exp(log_mel)

        # Least-squares un-projection from mel bands back to linear frequency.
        fb_pinv = torch.linalg.pinv(fb.to(torch.float32))
        power = torch.clamp(torch.matmul(fb_pinv, mel_power), min=0.0)
        magnitude = torch.sqrt(power)

        waveform = _griffin_lim(
            magnitude, self.n_fft, self.hop_length, window, n_iter=n_iter
        )
        if num_samples is not None:
            waveform = _fit_samples(waveform, num_samples)
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        waveform = waveform / peak * 0.95
        return waveform.squeeze(0) if squeeze else waveform


def _griffin_lim(magnitude: torch.Tensor, n_fft: int, hop_length: int,
                 window: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
    if _HAS_TORCHAUDIO:
        return taF.griffinlim(
            magnitude, window=window, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, power=1.0, n_iter=n_iter, momentum=0.99,
            length=None, rand_init=False,
        )

    angle = torch.zeros_like(magnitude)
    for _ in range(n_iter):
        spec = torch.polar(magnitude, angle)
        waveform = torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                               win_length=n_fft, window=window, center=True)
        rebuilt = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                             win_length=n_fft, window=window, center=True,
                             pad_mode="reflect", return_complex=True)
        angle = torch.angle(rebuilt)
    spec = torch.polar(magnitude, angle)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                       win_length=n_fft, window=window, center=True)


def _fit_frames(mel: torch.Tensor, n_frames: int) -> torch.Tensor:
    current = mel.shape[-1]
    if current == n_frames:
        return mel
    if current > n_frames:
        return mel[..., :n_frames]
    pad = mel.new_full(mel.shape[:-1] + (n_frames - current,), -1.0)
    return torch.cat([mel, pad], dim=-1)


def _fit_samples(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    current = waveform.shape[-1]
    if current == num_samples:
        return waveform
    if current > num_samples:
        return waveform[..., :num_samples]
    pad = waveform.new_zeros(waveform.shape[:-1] + (num_samples - current,))
    return torch.cat([waveform, pad], dim=-1)


def mel_center_frequencies(sample_rate: int, n_mels: int, f_min: float = 0.0,
                           f_max: Optional[float] = None) -> torch.Tensor:
    """Centre frequency (Hz) of each mel band -- lets a reward read pitch off a
    spectrogram without inverting it."""
    f_max = f_max if f_max is not None else sample_rate / 2
    mel_pts = torch.linspace(
        float(hz_to_mel(torch.tensor(f_min))),
        float(hz_to_mel(torch.tensor(f_max))),
        n_mels + 2,
    )
    return mel_to_hz(mel_pts)[1:-1]


def onset_envelope(mel: torch.Tensor) -> torch.Tensor:
    """Per-frame spectral flux: positive energy change summed over mel bands.

    Used by the AV-sync reward as a cheap stand-in for "when does something
    audibly happen".
    """
    energy = (mel + 1.0) / 2.0
    diff = energy[..., 1:] - energy[..., :-1]
    flux = torch.clamp(diff, min=0.0).sum(dim=-2)
    flux = torch.cat([flux[..., :1], flux], dim=-1)
    return flux
