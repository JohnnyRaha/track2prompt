"""Core audio analysis: globals, rhythm, harmony, production fingerprint."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import librosa

warnings.filterwarnings("ignore", category=UserWarning)

SR = 22050
HOP = 512

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Track:
    """A loaded audio file, mono for analysis plus stereo for imaging."""
    y: np.ndarray               # mono
    sr: int
    stereo: np.ndarray | None   # (2, n) if the source was stereo
    name: str = ""

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr


def load(path: str, sr: int = SR) -> Track:
    """Load a file to mono for analysis, keeping the stereo pair for imaging."""
    stereo_raw, native_sr = librosa.load(path, sr=sr, mono=False)
    if stereo_raw.ndim == 1:
        mono, stereo = stereo_raw, None
    else:
        stereo = stereo_raw
        mono = librosa.to_mono(stereo_raw)
    mono = mono.astype(np.float32)
    peak = np.max(np.abs(mono)) or 1.0
    return Track(y=mono / peak * 0.99, sr=native_sr, stereo=stereo,
                 name=path.rsplit("/", 1)[-1])


# --------------------------------------------------------------------------
# Tempo, metre, rhythm
# --------------------------------------------------------------------------

def _scalar(x) -> float:
    return float(np.atleast_1d(np.asarray(x)).ravel()[0])


def analyse_rhythm(t: Track) -> dict[str, Any]:
    onset_env = librosa.onset.onset_strength(y=t.y, sr=t.sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=t.sr,
                                           hop_length=HOP, trim=False)
    tempo = _scalar(tempo)
    beat_times = librosa.frames_to_time(beats, sr=t.sr, hop_length=HOP)

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=t.sr,
                                        hop_length=HOP, units="time")
    onset_rate = len(onsets) / max(t.duration, 1e-9)

    # Swing: where do off-beat onsets actually land between consecutive beats?
    swing = _estimate_swing(onsets, beat_times)

    # Pulse clarity — how periodic the onset envelope is
    ac = librosa.autocorrelate(onset_env, max_size=int(4 * t.sr / HOP))
    ac = ac / (ac[0] + 1e-9)
    pulse_clarity = float(np.max(ac[int(0.25 * t.sr / HOP):])) if len(ac) > 20 else 0.0

    # Beat-to-beat interval stability: low = programmed, high = played
    ibis = np.diff(beat_times)
    timing_var = float(np.std(ibis) / (np.mean(ibis) + 1e-9)) if len(ibis) > 2 else 0.0

    return {
        "tempo_bpm": round(tempo, 1),
        "beat_count": int(len(beat_times)),
        "onset_rate_per_sec": round(float(onset_rate), 2),
        "swing_ratio": round(swing, 3) if swing is not None else None,
        "pulse_clarity": round(pulse_clarity, 3),
        "timing_variability": round(timing_var, 4),
        "feel": _describe_feel(swing, timing_var),
    }


def _estimate_swing(onsets: np.ndarray, beats: np.ndarray) -> float | None:
    """Median normalised position of the strongest off-beat event.

    0.5 => straight eighths, ~0.66 => triplet swing.
    """
    if len(beats) < 4 or len(onsets) == 0:
        return None
    positions = []
    for a, b in zip(beats[:-1], beats[1:]):
        span = b - a
        if span <= 0:
            continue
        inner = onsets[(onsets > a + 0.15 * span) & (onsets < b - 0.15 * span)]
        if len(inner):
            positions.append(float((inner[0] - a) / span))
    if len(positions) < 4:
        return None
    return float(np.median(positions))


def _describe_feel(swing: float | None, timing_var: float) -> str:
    parts = []
    if swing is None:
        parts.append("sparse off-beat activity")
    elif swing > 0.60:
        parts.append("swung")
    elif swing > 0.55:
        parts.append("lightly shuffled")
    else:
        parts.append("straight")
    parts.append("tightly quantised" if timing_var < 0.02 else
                 "loosely played" if timing_var > 0.06 else "natural timing")
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Key and harmony
# --------------------------------------------------------------------------

def analyse_harmony(t: Track) -> dict[str, Any]:
    harmonic = librosa.effects.harmonic(t.y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=t.sr, hop_length=HOP)
    mean_chroma = chroma.mean(axis=1)
    mean_chroma = mean_chroma / (mean_chroma.sum() + 1e-9)

    # Bass-register chroma helps break the relative major/minor tie: the tonic
    # is usually the more prominent root in the low end.
    bass = librosa.effects.harmonic(
        librosa.effects.preemphasis(t.y, coef=-0.95), margin=2.0)
    bass_chroma = librosa.feature.chroma_cqt(
        y=bass, sr=t.sr, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
        n_octaves=3).mean(axis=1)
    bass_chroma = bass_chroma / (bass_chroma.sum() + 1e-9)

    key, mode, confidence, alt = _estimate_key(mean_chroma, bass_chroma)

    # Aggregate chroma over beats before matching chords. Frame-level matching
    # flickers badly on real mixes and reports a chord change every beat.
    _, beats = librosa.beat.beat_track(y=t.y, sr=t.sr, hop_length=HOP, trim=False)
    chords = _estimate_chords(chroma, t.sr, beats)

    return {
        "key": key,
        "mode": mode,
        "key_confidence": round(confidence, 3),
        "key_alternative": alt,
        "chord_sequence": chords["sequence"][:32],
        "unique_chords": chords["unique"],
        "harmonic_rhythm_sec": chords["mean_duration"],
        "harmonic_rhythm_beats": chords.get("mean_duration_beats"),
        "chord_count": chords.get("chord_count"),
        "chroma_profile": [round(float(v), 4) for v in mean_chroma],
    }


def _estimate_key(profile: np.ndarray,
                  bass_profile: np.ndarray | None = None
                  ) -> tuple[str, str, float, str | None]:
    """Krumhansl-Kessler correlation, with a relative-key tiebreak.

    C major and A minor contain identical pitch classes, so correlation alone
    picks between them almost at random. When the top two candidates are
    relatives and the margin is small, prefer whichever tonic is stronger in
    the bass register.
    """
    scores = []
    for i in range(12):
        rot = np.roll(profile, -i)
        scores.append((float(np.corrcoef(rot, KS_MAJOR)[0, 1]), PITCH_CLASSES[i], "major"))
        scores.append((float(np.corrcoef(rot, KS_MINOR)[0, 1]), PITCH_CLASSES[i], "minor"))
    scores.sort(reverse=True, key=lambda s: s[0])

    best, second = scores[0], scores[1]
    margin = best[0] - second[0]

    if bass_profile is not None and margin < 0.15 and _are_relatives(best, second):
        b_root = PITCH_CLASSES.index(best[1])
        s_root = PITCH_CLASSES.index(second[1])
        bass_margin = bass_profile[s_root] / (bass_profile[b_root] + 1e-9)
        if bass_margin > 1.15:
            best, second = second, best
            # Confidence now comes from the bass evidence, not the correlation
            # tie. Map the bass ratio onto a comparable 0-0.3 scale.
            margin = float(min(0.30, (bass_margin - 1.0) * 0.30))

    alt = f"{second[1]} {second[2]}" if _are_relatives(best, second) else None
    return best[1], best[2], float(margin), alt


def _are_relatives(a: tuple, b: tuple) -> bool:
    """True if a and b are relative major/minor of each other."""
    if a[2] == b[2]:
        return False
    maj, mn = (a, b) if a[2] == "major" else (b, a)
    return (PITCH_CLASSES.index(maj[1]) - PITCH_CLASSES.index(mn[1])) % 12 == 3


def _chord_templates() -> tuple[np.ndarray, list[str]]:
    templates, names = [], []
    for i, root in enumerate(PITCH_CLASSES):
        maj = np.zeros(12); maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1
        mn = np.zeros(12);  mn[[i, (i + 3) % 12, (i + 7) % 12]] = 1
        templates += [maj / maj.sum(), mn / mn.sum()]
        names += [f"{root}", f"{root}m"]
    return np.array(templates), names


def _estimate_chords(chroma: np.ndarray, sr: int,
                     beats: np.ndarray | None = None) -> dict[str, Any]:
    """Template-match chords over beat-synchronous chroma.

    Matching frame by frame produces a new "chord" several times a second on any
    real mix, because passing tones and percussion move the chroma around.
    Averaging chroma within each beat removes almost all of that.
    """
    templates, names = _chord_templates()

    if beats is not None and len(beats) > 4:
        agg = librosa.util.sync(chroma, beats, aggregate=np.median)
        unit_sec = float(np.median(np.diff(
            librosa.frames_to_time(beats, sr=sr, hop_length=HOP))))
    else:
        agg = chroma
        unit_sec = HOP / sr

    norm = agg / (agg.sum(axis=0, keepdims=True) + 1e-9)
    scores = templates @ norm
    idx = np.argmax(scores, axis=0)

    # Smooth across beats: a chord lasting under two beats is usually an artefact
    from scipy.signal import medfilt
    if len(idx) >= 5:
        idx = medfilt(idx.astype(float), kernel_size=3).astype(int)

    seq, runs = [], []
    cur, count = idx[0], 1
    for v in idx[1:]:
        if v == cur:
            count += 1
        else:
            seq.append(names[cur]); runs.append(count)
            cur, count = v, 1
    seq.append(names[cur]); runs.append(count)

    durations = [r * unit_sec for r in runs]
    min_dur = unit_sec * 1.5
    keep = [(c, d) for c, d in zip(seq, durations) if d >= min_dur]
    if not keep:
        keep = list(zip(seq, durations))

    # Rank by total time, not by appearance - that is what "the chords" means
    totals: dict[str, float] = {}
    for c, d in keep:
        totals[c] = totals.get(c, 0.0) + d
    ranked = sorted(totals, key=lambda c: -totals[c])

    return {
        "sequence": [c for c, _ in keep],
        "unique": ranked[:8],
        "chord_count": len(totals),
        "mean_duration": round(float(np.mean([d for _, d in keep])), 2),
        "mean_duration_beats": round(
            float(np.mean([d for _, d in keep]) / unit_sec), 1) if unit_sec else None,
    }


# --------------------------------------------------------------------------
# Production fingerprint
# --------------------------------------------------------------------------

def analyse_production(t: Track) -> dict[str, Any]:
    S = np.abs(librosa.stft(t.y, hop_length=HOP))
    centroid = librosa.feature.spectral_centroid(S=S, sr=t.sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=t.sr, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    rms = librosa.feature.rms(S=S)[0]

    peak = float(np.max(np.abs(t.y)))
    rms_overall = float(np.sqrt(np.mean(t.y ** 2)))
    crest_db = 20 * np.log10((peak + 1e-9) / (rms_overall + 1e-9))

    # Loudness proxy (not true LUFS; use pyloudnorm if you need the real thing)
    loudness_db = 20 * np.log10(rms_overall + 1e-9)

    # Dynamic range: spread of frame loudness
    frame_db = 20 * np.log10(rms + 1e-9)
    dr_db = float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10))

    bands = _band_energy(S, t.sr)
    width = _stereo_width(t)
    rt60 = _reverb_estimate(t)
    transient = _transient_sharpness(t)

    return {
        "spectral_centroid_hz": round(float(np.mean(centroid)), 1),
        "spectral_rolloff95_hz": round(float(np.mean(rolloff)), 1),
        "spectral_flatness": round(float(np.mean(flatness)), 4),
        "crest_factor_db": round(float(crest_db), 2),
        "loudness_dbfs_rms": round(float(loudness_db), 2),
        "dynamic_range_db": round(dr_db, 2),
        "band_energy": bands,
        "stereo_width": width,
        "reverb_decay_sec": rt60,
        "transient_sharpness": transient,
        "brightness": _label_brightness(float(np.mean(centroid))),
        "compression": _label_compression(crest_db, dr_db),
    }


def _band_energy(S: np.ndarray, sr: int) -> dict[str, float]:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
    power = (S ** 2).mean(axis=1)
    total = power.sum() + 1e-9
    edges = {"sub": (0, 60), "bass": (60, 250), "low_mid": (250, 800),
             "mid": (800, 2500), "high_mid": (2500, 6000), "air": (6000, sr / 2)}
    return {k: round(float(power[(freqs >= lo) & (freqs < hi)].sum() / total), 4)
            for k, (lo, hi) in edges.items()}


def _stereo_width(t: Track) -> dict[str, Any]:
    if t.stereo is None or t.stereo.shape[0] < 2:
        return {"is_stereo": False, "side_mid_ratio": 0.0, "label": "mono"}
    L, R = t.stereo[0], t.stereo[1]
    mid = (L + R) / 2
    side = (L - R) / 2
    m = float(np.sqrt(np.mean(mid ** 2))) + 1e-9
    s = float(np.sqrt(np.mean(side ** 2)))
    ratio = s / m
    label = ("near-mono" if ratio < 0.05 else "narrow" if ratio < 0.2 else
             "moderately wide" if ratio < 0.45 else "very wide")
    return {"is_stereo": True, "side_mid_ratio": round(ratio, 4), "label": label}


def _reverb_estimate(t: Track) -> float | None:
    """Crude RT60 proxy: median decay slope after the loudest onsets."""
    env = librosa.feature.rms(y=t.y, hop_length=HOP)[0]
    db = 20 * np.log10(env + 1e-9)
    onsets = librosa.onset.onset_detect(y=t.y, sr=t.sr, hop_length=HOP, units="frames")
    if len(onsets) < 5:
        return None
    slopes = []
    for o in onsets:
        seg = db[o:o + 40]
        if len(seg) < 15:
            continue
        drop = seg[0] - seg.min()
        if drop < 6:
            continue
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]          # dB per frame
        if slope < -0.05:
            slopes.append(slope)
    if not slopes:
        return None
    per_sec = float(np.median(slopes)) * (t.sr / HOP)
    return round(float(-60.0 / per_sec), 2) if per_sec < 0 else None


def _transient_sharpness(t: Track) -> float:
    env = librosa.onset.onset_strength(y=t.y, sr=t.sr, hop_length=HOP)
    if env.std() == 0:
        return 0.0
    return round(float(np.percentile(env, 98) / (np.mean(env) + 1e-9)), 2)


def _label_brightness(centroid: float) -> str:
    if centroid < 1200: return "dark"
    if centroid < 2000: return "warm"
    if centroid < 3200: return "balanced"
    if centroid < 4500: return "bright"
    return "very bright"


def _label_compression(crest_db: float, dr_db: float) -> str:
    if crest_db < 8 or dr_db < 6:  return "heavily compressed, loud master"
    if crest_db < 12 or dr_db < 10: return "moderately compressed"
    if crest_db < 16:               return "lightly compressed"
    return "dynamic, open master"
