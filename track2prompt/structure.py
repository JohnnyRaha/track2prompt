"""Song structure: self-similarity, novelty boundaries, repeated-section labelling."""

from __future__ import annotations

from typing import Any

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import AgglomerativeClustering

from .analysis import Track, HOP


def analyse_structure(t: Track, max_sections: int = 8) -> dict[str, Any]:
    """Segment the track and label repeated sections with the same letter."""
    feats = _features(t)
    if feats.shape[1] < 20:
        return {"sections": [], "form": "", "note": "track too short to segment"}

    # Minimum musical section length, scaled to the track. A 4-minute song does
    # not have 17 sections; a fixed 4-second floor guarantees over-segmentation.
    min_sec = max(8.0, min(20.0, t.duration / 14.0))

    # Detect boundaries at finer resolution than the final minimum section
    # length, then enforce the minimum afterwards. Using min_sec for both makes
    # peak picking suppress real boundaries on short tracks.
    ssm = _self_similarity(feats)
    bounds = _novelty_boundaries(ssm,
                                 min_gap_frames=_sec_to_frames(t, min_sec * 0.45))
    bounds = _with_edges(bounds, feats.shape[1])
    bounds = _enforce_min_length(bounds, _sec_to_frames(t, min_sec))

    if len(bounds) < 3:
        return {"sections": [], "form": "", "note": "no clear boundaries found"}

    labels = _label_segments(feats, bounds, max_sections)
    bounds, labels = _merge_runs(bounds, labels,
                                 max_frames=_sec_to_frames(t, max(45.0, t.duration / 6)))
    sections = _build_sections(t, bounds, labels)
    sections = _name_sections(sections, t.duration)

    return {
        "sections": sections,
        "form": "-".join(s["label"] for s in sections),
        "section_count": len(sections),
        "repetition_ratio": round(1 - len({s["label"] for s in sections}) / len(sections), 3),
    }


# --------------------------------------------------------------------------

def _sec_to_frames(t: Track, seconds: float) -> int:
    return max(1, int(seconds * t.sr / HOP))


def _features(t: Track) -> np.ndarray:
    """Stack beat-agnostic timbral + harmonic features, L2-normalised."""
    chroma = librosa.feature.chroma_cqt(y=t.y, sr=t.sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=t.y, sr=t.sr, hop_length=HOP, n_mfcc=13)
    rms = librosa.feature.rms(y=t.y, hop_length=HOP)

    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    stack = np.vstack([
        _z(chroma[:, :n]) * 1.0,
        _z(mfcc[1:, :n]) * 1.0,        # drop MFCC0 (loudness)
        _z(rms[:, :n]) * 0.5,
    ])
    # Smooth to emphasise section-scale change over frame-scale change
    stack = gaussian_filter1d(stack, sigma=8, axis=1)
    return stack / (np.linalg.norm(stack, axis=0, keepdims=True) + 1e-9)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)


def _self_similarity(feats: np.ndarray) -> np.ndarray:
    return feats.T @ feats


def _novelty_boundaries(ssm: np.ndarray, min_gap_frames: int,
                        kernel: int = 64) -> list[int]:
    """Foote checkerboard novelty along the SSM diagonal."""
    k = _checkerboard(kernel)
    half = kernel // 2
    n = ssm.shape[0]
    novelty = np.zeros(n)
    for i in range(half, n - half):
        novelty[i] = float(np.sum(ssm[i - half:i + half, i - half:i + half] * k))
    novelty = gaussian_filter1d(novelty, sigma=4)
    if novelty.max() > novelty.min():
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())

    peaks = librosa.util.peak_pick(
        novelty, pre_max=min_gap_frames, post_max=min_gap_frames,
        pre_avg=min_gap_frames, post_avg=min_gap_frames,
        delta=0.08, wait=min_gap_frames,
    )
    return sorted(int(p) for p in peaks)


def _checkerboard(size: int) -> np.ndarray:
    half = size // 2
    k = np.ones((size, size))
    k[:half, half:] = -1
    k[half:, :half] = -1
    g = np.outer(*(2 * [np.exp(-0.5 * (np.linspace(-2, 2, size) ** 2))]))
    return k * g


def _with_edges(bounds: list[int], n_frames: int) -> list[int]:
    out = [0] + [b for b in bounds if 0 < b < n_frames] + [n_frames]
    return sorted(set(out))


def _enforce_min_length(bounds: list[int], min_frames: int) -> list[int]:
    """Drop boundaries that would create a section shorter than min_frames."""
    if len(bounds) < 3:
        return bounds
    out = [bounds[0]]
    for b in bounds[1:-1]:
        if b - out[-1] >= min_frames:
            out.append(b)
    # Never leave a runt final section - fold it back into the previous one
    if bounds[-1] - out[-1] < min_frames and len(out) > 1:
        out.pop()
    out.append(bounds[-1])
    return out


def _label_segments(feats: np.ndarray, bounds: list[int], max_sections: int) -> list[int]:
    """Cluster segment centroids so musically similar sections share a label."""
    centroids = np.array([feats[:, a:b].mean(axis=1)
                          for a, b in zip(bounds[:-1], bounds[1:])])
    n_seg = len(centroids)
    # Most songs have 3-6 distinct section types regardless of how many times
    # each recurs. Allowing one cluster per two segments invents distinctions.
    n_clusters = max(1, min(max_sections, n_seg, max(2, round(n_seg / 2.5))))
    if n_seg < 2:
        return [0] * n_seg
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average",
                                    metric="cosine")
    return list(model.fit_predict(centroids))


def _merge_runs(bounds: list[int], labels: list[int],
                max_frames: int | None = None) -> tuple[list[int], list[int]]:
    """Collapse consecutive segments sharing a label, up to a length ceiling.

    Novelty detection splits an eight-bar section at its four-bar midpoint, and
    merging same-label neighbours fixes that. But on loop-based material where
    every section shares a timbral palette, unchecked merging swallows the whole
    track into two blocks. `max_frames` stops a merged run growing past a length
    no real section reaches.
    """
    if not labels:
        return bounds, labels
    new_bounds = [bounds[0]]
    new_labels = [labels[0]]
    for i in range(1, len(labels)):
        same = labels[i] == new_labels[-1]
        would_span = bounds[i + 1] - new_bounds[-1] if i + 1 < len(bounds) else 0
        too_long = max_frames is not None and would_span > max_frames
        if same and not too_long:
            continue                      # extend the current run
        new_bounds.append(bounds[i])
        new_labels.append(labels[i])
    new_bounds.append(bounds[-1])
    return new_bounds, new_labels


def _build_sections(t: Track, bounds: list[int], labels: list[int]) -> list[dict]:
    letters = {}
    out = []
    for (a, b), lab in zip(zip(bounds[:-1], bounds[1:]), labels):
        if lab not in letters:
            letters[lab] = chr(ord("A") + len(letters))
        start = librosa.frames_to_time(a, sr=t.sr, hop_length=HOP)
        end = librosa.frames_to_time(b, sr=t.sr, hop_length=HOP)
        seg = t.y[int(start * t.sr):int(end * t.sr)]
        out.append({
            "label": letters[lab],
            "start_sec": round(float(start), 2),
            "end_sec": round(float(end), 2),
            "duration_sec": round(float(end - start), 2),
            "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)), 2)
            if len(seg) else None,
        })
    return out


def _name_sections(sections: list[dict], total: float) -> list[dict]:
    """Attach a plain-language guess at each section's role.

    Uses each section's rank among the others rather than an absolute dB
    threshold. On a compressed master every section sits within a couple of dB
    of the loudest, so a fixed threshold labels almost everything a chorus.
    """
    if not sections:
        return sections
    levels = [s["rms_db"] for s in sections if s["rms_db"] is not None]
    if not levels:
        return sections

    order = sorted(levels)
    def pct(db: float) -> float:
        return sum(1 for v in order if v <= db) / len(order)

    n = len(sections)
    for i, s in enumerate(sections):
        db = s["rms_db"]
        p = pct(db) if db is not None else 0.5
        first, last = i == 0, i == n - 1

        if first and p <= 0.4:
            role = "intro"
        elif last and p <= 0.5:
            role = "outro"
        elif p >= 0.8:
            role = "chorus / peak section"
        elif p <= 0.25:
            role = "breakdown / quieter section"
        else:
            role = "verse / development"
        s["role"] = role
        s["level_rank"] = round(p, 2)
    return sections
