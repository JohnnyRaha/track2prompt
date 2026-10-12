"""Suno-specific output.

Suno splits input across fields that behave differently from a single prose
prompt:

  Style field   - comma-separated tags, NOT sentences. Earlier tags are
                  weighted more heavily. 1,000 chars on v4.5/v5/v5.5,
                  200 on v4 and older. Truncation is silent.
  Lyrics field  - singable text plus bracketed section tags. Structure lives
                  here, never in the style field.
  Negative tags - up to 200 chars of things to steer away from.

Brackets in the style field are a common mistake: they are section-tag syntax
and belong in the lyrics field only. BPM and key go in the style field as bare
text.
"""

from __future__ import annotations

from typing import Any

# Field caps by model generation
LIMITS = {
    "v5.5": {"style": 1000, "lyrics": 5000, "title": 100, "negative": 200},
    "v5":   {"style": 1000, "lyrics": 5000, "title": 100, "negative": 200},
    "v4.5": {"style": 1000, "lyrics": 5000, "title": 100, "negative": 200},
    "v4":   {"style": 200,  "lyrics": 3000, "title": 80,  "negative": 200},
}

# Even where 1,000 chars are allowed, dilution sets in. Keep the high-signal
# tags inside this budget and let the rest be optional colour.
CORE_TAG_BUDGET = 180

ROLE_TO_TAG = {
    "intro": "Intro",
    "verse / development": "Verse",
    "chorus / peak section": "Chorus",
    "breakdown / quieter section": "Break",
    "breakdown": "Break",
    "outro": "Outro",
}


def build_suno(analysis: dict[str, Any], style_hint: str | None = None,
               model: str = "v5", instrumental: bool | None = None,
               title: str | None = None) -> dict[str, Any]:
    """Produce Suno-ready fields from a track2prompt analysis."""
    limits = LIMITS.get(model, LIMITS["v5"])

    tags = _style_tags(analysis, style_hint, instrumental)
    style, dropped = _fit(tags, min(limits["style"], CORE_TAG_BUDGET * 3))
    core, _ = _fit(tags, CORE_TAG_BUDGET)

    structure = _structure_block(analysis, instrumental)
    negative = _negative_tags(analysis)

    return {
        "model": model,
        "style": style,
        "style_core": core,
        "style_chars": len(style),
        "style_limit": limits["style"],
        "dropped_tags": dropped,
        "lyrics_field": structure,
        "lyrics_chars": len(structure),
        "negative_tags": negative,
        "title": (title or _suggest_title(analysis))[:limits["title"]],
        "notes": _notes(analysis, instrumental, model),
    }


# --------------------------------------------------------------------------
# Style tags, in Suno's preferred weighting order
# --------------------------------------------------------------------------

def _style_tags(a: dict[str, Any], hint: str | None,
                instrumental: bool | None) -> list[str]:
    r = a.get("rhythm", {})
    h = a.get("harmony", {})
    p = a.get("production", {})
    bands = p.get("band_energy", {})

    tags: list[str] = []

    # 1. Genre first - Suno weights the opening most heavily
    if hint:
        tags += [t.strip() for t in hint.split(",") if t.strip()]

    # 2. Vocal direction. Suno picks a random voice if you leave this out.
    if instrumental is True:
        tags.append("instrumental")
    elif instrumental is False:
        tags.append("vocals")

    # 3. Mood / energy
    tags.append(_mood(p, r))

    # 4. Instrumentation cues, inferred from spectral balance only
    tags += _instrument_cues(bands, p)

    # 5. Rhythm feel
    feel = r.get("feel", "")
    if "swung" in feel:
        tags.append("swung groove")
    elif "shuffled" in feel:
        tags.append("shuffled groove")
    if "loosely played" in feel:
        tags.append("loose live feel")
    elif "tightly quantised" in feel:
        tags.append("tight programmed groove")

    # 6. Production character
    comp = p.get("compression", "")
    if "heavily" in comp:
        tags.append("loud compressed master")
    elif "dynamic" in comp:
        tags.append("dynamic open mix")

    rt60 = p.get("reverb_decay_sec")
    if rt60 is not None:
        tags.append("dry close mix" if rt60 < 0.4 else
                    "roomy reverb" if rt60 < 1.8 else "spacious reverb")

    width = (p.get("stereo_width") or {}).get("label")
    if width == "very wide":
        tags.append("wide stereo")
    elif width in ("narrow", "near-mono"):
        tags.append("narrow focused mix")

    b = p.get("brightness")
    if b in ("dark", "warm"):
        tags.append(f"{b} tone")
    elif b in ("bright", "very bright"):
        tags.append("bright tone")

    # 7. Tempo and key last - bare text, never in brackets
    bpm = r.get("tempo_bpm")
    if bpm:
        tags.append(f"{bpm:.0f} BPM")
    if h.get("key"):
        tags.append(f"{h['key']} {h.get('mode', '')}".strip())

    # De-duplicate, preserving order
    seen, out = set(), []
    for t in tags:
        k = t.lower()
        if k and k not in seen:
            seen.add(k); out.append(t)
    return out


def _mood(p: dict, r: dict) -> str:
    bright = p.get("brightness", "balanced")
    bpm = r.get("tempo_bpm", 120)
    if bright in ("dark", "warm") and bpm < 110:
        return "mellow"
    if bright in ("bright", "very bright") and bpm >= 125:
        return "energetic"
    if bpm < 95:
        return "laid back"
    return "steady driving"


def _instrument_cues(bands: dict, p: dict) -> list[str]:
    """Timbral hints only. The analyser does not identify instruments, so
    describe the spectrum rather than naming things that may not be there."""
    out = []
    sub = bands.get("sub", 0) + bands.get("bass", 0)
    air = bands.get("air", 0)
    if sub > 0.5:
        out.append("deep bass")
    elif sub > 0.35:
        out.append("warm bass")
    if air > 0.12:
        out.append("crisp highs")
    if p.get("transient_sharpness", 4) > 6:
        out.append("punchy drums")
    elif p.get("transient_sharpness", 4) < 3:
        out.append("soft drums")
    return out


def _fit(tags: list[str], budget: int) -> tuple[str, list[str]]:
    """Join tags up to a character budget, dropping from the end."""
    kept, dropped, total = [], [], 0
    for t in tags:
        add = len(t) + (2 if kept else 0)
        if total + add <= budget:
            kept.append(t); total += add
        else:
            dropped.append(t)
    return ", ".join(kept), dropped


# --------------------------------------------------------------------------
# Structure -> lyrics-field metatags
# --------------------------------------------------------------------------

def _structure_block(a: dict[str, Any], instrumental: bool | None) -> str:
    sections = (a.get("structure") or {}).get("sections") or []
    if not sections:
        return "[Instrumental]" if instrumental else ""

    lines = []
    for s in sections:
        tag = ROLE_TO_TAG.get(s.get("role", ""), "Verse")
        lines.append(f"[{tag}]")
        if instrumental:
            lines.append(f"[Instrumental - {s['duration_sec']:.0f}s]")
        lines.append("")
    return "\n".join(lines).strip()


def _negative_tags(a: dict[str, Any]) -> str:
    """Steer away from the obvious opposite of what was measured."""
    p = a.get("production", {})
    out = []
    b = p.get("brightness")
    if b in ("dark", "warm"):
        out.append("harsh treble")
    elif b in ("bright", "very bright"):
        out.append("muddy")
    comp = p.get("compression", "")
    if "dynamic" in comp:
        out.append("overcompressed")
    if (p.get("stereo_width") or {}).get("label") in ("narrow", "near-mono"):
        out.append("wide stereo")
    out.append("distorted")
    return ", ".join(out)[:200]


def _suggest_title(a: dict[str, Any]) -> str:
    src = a.get("source", {}).get("file", "Untitled")
    return src.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip().title()


def _notes(a: dict[str, Any], instrumental: bool | None, model: str) -> list[str]:
    notes = []
    if instrumental is None:
        notes.append("Vocal presence was not specified. Suno will choose a voice "
                     "at random unless you pass --instrumental or --vocals.")
    h = a.get("harmony", {})
    if h.get("key_confidence", 1) < 0.08 and h.get("key_alternative"):
        notes.append(f"Key is ambiguous between {h['key']} {h['mode']} and "
                     f"{h['key_alternative']}. Suno mostly ignores key anyway; "
                     f"drop it from the style field if output drifts.")
    if model == "v4":
        notes.append("v4 caps the style field at 200 characters and truncates "
                     "silently. Use style_core.")
    if not (a.get("structure") or {}).get("sections"):
        notes.append("No sections detected, so the lyrics field is empty. "
                     "Add your own [Verse]/[Chorus] tags if you want structure.")
    return notes


def format_suno(out: dict[str, Any]) -> str:
    """Human-readable block for the terminal."""
    L = []
    L.append(f"SUNO  ({out['model']})")
    L.append("=" * 64)
    L.append(f"\nSTYLE FIELD  [{out['style_chars']}/{out['style_limit']} chars]\n")
    L.append(out["style"])
    if out["style_core"] != out["style"]:
        L.append(f"\nSTYLE, TRIMMED  [{len(out['style_core'])} chars - use on v4]\n")
        L.append(out["style_core"])
    if out["lyrics_field"]:
        L.append(f"\nLYRICS FIELD  [{out['lyrics_chars']} chars]\n")
        L.append(out["lyrics_field"])
    L.append(f"\nNEGATIVE TAGS\n\n{out['negative_tags']}")
    L.append(f"\nTITLE\n\n{out['title']}")
    if out["notes"]:
        L.append("\nNOTES\n")
        for n in out["notes"]:
            L.append(f"  - {n}")
    return "\n".join(L)
