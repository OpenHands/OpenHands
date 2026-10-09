"""Original ambient background music, synthesized from scratch (no samples).

Usage: python3 make_music.py OUT.wav SECONDS
A slow Cmaj7 - Am9 - Fmaj7 - G6sus pad with a sine bass, sparse bell notes
and a synthetic convolution reverb. Deterministic (fixed seed).
"""

import sys
import wave

import numpy as np

SR = 44100
rng = np.random.default_rng(18200)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def envelope(n, attack, release):
    env = np.ones(n)
    a = min(n, int(attack * SR))
    r = min(n - a, int(release * SR))
    env[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if r > 0:
        env[n - r :] = np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return env


def pad_voice(freq, n):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for cents in (-6.0, 0.0, 6.0):
        f = freq * 2 ** (cents / 1200)
        phase = rng.uniform(0, 2 * np.pi)
        # Soft saw: harmonics rolled off steeply (warm, no buzz).
        for h in range(1, 7):
            out += np.sin(2 * np.pi * f * h * t + phase * h) / h**2.2
    # Slow shimmer.
    lfo = 1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.07, 0.13) * t + rng.uniform(0, 6))
    return out * lfo / 3


def bell(freq, n):
    t = np.arange(n) / SR
    tone = (
        np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * freq * 2.0 * t)
        + 0.12 * np.sin(2 * np.pi * freq * 3.01 * t)
    )
    env = np.exp(-t * 2.2) * (1 - np.exp(-t * 300))
    return tone * env


def reverb(x, seconds=3.2, mix=0.32):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 3.0 / seconds * 2.3)
    # Darken the tail: one-pole lowpass via cumulative smoothing.
    ir = np.convolve(ir, np.ones(24) / 24, mode="same")
    ir /= np.sqrt(np.sum(ir**2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    wet *= np.std(x) / (np.std(wet) + 1e-12)
    return (1 - mix) * x + mix * wet


def main(out, seconds):
    total = int(seconds * SR)
    mix = np.zeros(total)
    bar = 4.8  # seconds per chord (~50 BPM feel)
    chords = [
        (48, [60, 64, 67, 71]),  # Cmaj7
        (45, [60, 64, 67, 69, 71]),  # Am9 (voiced around C)
        (41, [60, 65, 69, 72]),  # Fmaj7
        (43, [62, 67, 69, 71]),  # G6sus-ish
    ]
    bells = [72, 76, 79, 74, 81, 79, 76, 72]
    i = 0
    start = 0.0
    while start < seconds:
        root, notes = chords[i % len(chords)]
        s = int(start * SR)
        n = min(total - s, int((bar + 2.4) * SR))  # overlap for crossfade
        env = envelope(n, 1.6, 2.4)
        chord = sum(pad_voice(midi_hz(m), n) for m in notes) / len(notes)
        bass = np.sin(2 * np.pi * midi_hz(root) * np.arange(n) / SR)
        mix[s : s + n] += env * (0.55 * chord + 0.22 * bass)
        # Two sparse bell notes per bar.
        for k, offset in enumerate((0.6, 2.9)):
            bs = s + int(offset * SR)
            if bs >= total:
                continue
            bn = min(total - bs, int(3.5 * SR))
            note = bells[(2 * i + k) % len(bells)]
            mix[bs : bs + bn] += 0.11 * bell(midi_hz(note), bn)
        start += bar
        i += 1
    mix = reverb(mix)
    mix *= envelope(total, 2.5, 3.5)
    mix /= np.max(np.abs(mix)) + 1e-9
    mix *= 0.89  # -1 dBFS peak; the final level is set in ffmpeg
    # Gentle stereo: slightly different reverb per side would cost twice;
    # a small Haas delay is enough for width.
    d = int(0.011 * SR)
    left = mix
    right = np.concatenate([np.zeros(d), mix[:-d]])
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
