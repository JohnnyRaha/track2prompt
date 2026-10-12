"""Build a synthetic multi-section track with known properties, to sanity-check
the analyser. Ground truth: 124 BPM, A minor, form intro-A-B-A-outro.
"""
import numpy as np, soundfile as sf

SR, BPM = 44100, 124.0
BEAT = 60.0 / BPM


def sine(f, n, sr=SR):
    return np.sin(2 * np.pi * f * np.arange(n) / sr)


def env(n, a=0.005, d=0.25):
    e = np.ones(n)
    na, nd = int(a * SR), int(d * SR)
    e[:na] = np.linspace(0, 1, na)
    e[na:na + nd] = np.linspace(1, 0.25, min(nd, n - na))
    e[na + nd:] = 0.25
    return e


def note(freq, dur, amp=0.3, harmonics=3):
    n = int(dur * SR)
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        out += (0.6 ** h) * sine(freq * h, n)
    return out * env(n) * amp


def kick(dur=0.18):
    n = int(dur * SR)
    f = np.linspace(110, 45, n)
    ph = np.cumsum(2 * np.pi * f / SR)
    return np.sin(ph) * np.exp(-np.linspace(0, 7, n)) * 0.85


def snare(dur=0.14):
    n = int(dur * SR)
    return (np.random.randn(n) * np.exp(-np.linspace(0, 16, n))) * 0.45


def hat(dur=0.05):
    n = int(dur * SR)
    return (np.random.randn(n) * np.exp(-np.linspace(0, 40, n))) * 0.2


NOTE = {"A2": 110.00, "C3": 130.81, "E3": 164.81, "G3": 196.00,
        "A3": 220.00, "C4": 261.63, "D4": 293.66, "E4": 329.63,
        "F4": 349.23, "G4": 392.00, "A4": 440.00}

# Am - F - C - G, four beats each
PROG = [("A3", ["A3", "C4", "E4"]), ("F4", ["F4", "A4", "C4"]),
        ("C4", ["C4", "E4", "G4"]), ("G4", ["G4", "D4", "A4"])]


def section(bars, drums=True, chords=True, bass=True, lead=False, gain=1.0):
    n = int(bars * 4 * BEAT * SR)
    buf = np.zeros(n)
    for bar in range(bars):
        root, triad = PROG[bar % 4]
        bar_start = int(bar * 4 * BEAT * SR)
        if chords:
            for pitch in triad:
                c = note(NOTE[pitch], 4 * BEAT, amp=0.10, harmonics=2)
                buf[bar_start:bar_start + len(c)] += c[:max(0, n - bar_start)]
        if bass:
            for b in range(4):
                s = bar_start + int(b * BEAT * SR)
                bn = note(NOTE[root] / 2, BEAT * 0.9, amp=0.28, harmonics=2)
                end = min(s + len(bn), n)
                buf[s:end] += bn[:end - s]
        if drums:
            for b in range(4):
                s = bar_start + int(b * BEAT * SR)
                k = kick(); end = min(s + len(k), n); buf[s:end] += k[:end - s]
                if b % 2 == 1:
                    sn = snare(); end = min(s + len(sn), n); buf[s:end] += sn[:end - s]
                for eighth in (0, 1):
                    hs = s + int(eighth * BEAT * 0.5 * SR)
                    hh = hat(); end = min(hs + len(hh), n); buf[hs:end] += hh[:end - hs]
        if lead:
            mel = ["E4", "G4", "A4", "G4"]
            for b in range(4):
                s = bar_start + int(b * BEAT * SR)
                ln = note(NOTE[mel[b]], BEAT * 0.8, amp=0.22, harmonics=4)
                end = min(s + len(ln), n); buf[s:end] += ln[:end - s]
    return buf * gain


parts = [
    section(4, drums=False, lead=False, gain=0.5),   # intro
    section(8, lead=False, gain=0.9),                # A
    section(8, lead=True,  gain=1.0),                # B (chorus, louder + lead)
    section(8, lead=False, gain=0.9),                # A
    section(4, drums=False, lead=False, gain=0.45),  # outro
]
mono = np.concatenate(parts)
mono /= np.max(np.abs(mono)) * 1.05

# Simple stereo: delay one side slightly for width
delay = int(0.004 * SR)
left = mono
right = np.concatenate([np.zeros(delay), mono[:-delay]])
stereo = np.stack([left, right], axis=1).astype(np.float32)

sf.write("test_track.wav", stereo, SR)
print(f"wrote test_track.wav  {len(mono)/SR:.1f}s  "
      f"ground truth: {BPM:.0f} BPM, A minor, intro-A-B-A-outro")
