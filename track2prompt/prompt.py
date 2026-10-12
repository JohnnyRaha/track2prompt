"""Turn an analysis dict into a music-model prompt.

Two paths:
  build_prompt()  - deterministic, rule-based, no dependencies. Always works.
  refine_prompt() - optional LLM pass to make it read naturally.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------
# Vocabulary mapping
# --------------------------------------------------------------------------

def _tempo_word(bpm: float) -> str:
    if bpm < 70:  return "slow"
    if bpm < 90:  return "downtempo"
    if bpm < 110: return "mid-tempo"
    if bpm < 130: return "upbeat"
    if bpm < 150: return "driving"
    return "fast"


def _density_word(rate: float) -> str:
    if rate < 1.5: return "sparse"
    if rate < 3.5: return "steady"
    if rate < 6.0: return "busy"
    return "dense"


def _energy_from_bands(bands: dict[str, float]) -> str:
    low = bands.get("sub", 0) + bands.get("bass", 0)
    high = bands.get("high_mid", 0) + bands.get("air", 0)
    if low > 0.5:  return "bass-heavy"
    if high > 0.3: return "top-end forward"
    return "midrange-centred"


def _reverb_word(rt60: float | None) -> str | None:
    if rt60 is None: return None
    if rt60 < 0.3:  return "dry, close-miked space"
    if rt60 < 0.8:  return "small room ambience"
    if rt60 < 1.8:  return "roomy, medium reverb"
    return "large reverberant space"


def _transient_word(sharp: float) -> str:
    if sharp < 3:  return "soft, rounded transients"
    if sharp < 6:  return "natural attack"
    return "sharp, punchy transients"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def build_prompt(a: dict[str, Any], style_hint: str | None = None,
                 include_structure: bool = True) -> dict[str, str]:
    """Return short / standard / detailed prompt variants."""
    r = a.get("rhythm", {})
    h = a.get("harmony", {})
    p = a.get("production", {})
    s = a.get("structure", {})

    bpm = r.get("tempo_bpm", 120)
    key = f"{h.get('key', 'C')} {h.get('mode', 'major')}"

    core: list[str] = []
    if style_hint:
        core.append(style_hint.strip())
    core.append(f"{_tempo_word(bpm)} at {bpm:.0f} BPM")
    core.append(f"in {key}")

    feel = r.get("feel")
    if feel:
        core.append(feel)

    texture: list[str] = []
    texture.append(_density_word(r.get("onset_rate_per_sec", 3)) + " rhythmic activity")
    texture.append(_energy_from_bands(p.get("band_energy", {})))
    if p.get("brightness"):
        b = p["brightness"]
        texture.append(b if b == "balanced" else f"{b} tonal balance")

    production: list[str] = []
    if p.get("compression"):
        production.append(p["compression"])
    rv = _reverb_word(p.get("reverb_decay_sec"))
    if rv:
        production.append(rv)
    production.append(_transient_word(p.get("transient_sharpness", 4)))
    width = (p.get("stereo_width") or {}).get("label")
    if width and width != "mono":
        production.append(f"{width} stereo image")

    harmony_bits: list[str] = []
    chords = h.get("unique_chords") or []
    if 1 < len(chords) <= 8:
        total = h.get("chord_count", len(chords))
        lead = "chords drawn from " if total <= len(chords) else "mainly built on "
        harmony_bits.append(lead + ", ".join(chords))
    beats = h.get("harmonic_rhythm_beats")
    if beats and beats >= 1.5:
        bars = beats / 4.0
        if bars >= 0.9:
            harmony_bits.append(f"chord changes about every {bars:.0f} bar"
                                f"{'s' if bars >= 1.8 else ''}")
        else:
            harmony_bits.append(f"chord changes every {beats:.0f} beats")

    short = ", ".join(core[:3]) + ". " + ", ".join(texture[:2]) + "."

    standard_parts = [", ".join(core) + ".",
                      ", ".join(texture) + ".",
                      ", ".join(production) + "."]
    standard = " ".join(standard_parts)

    detailed_parts = list(standard_parts)
    if harmony_bits:
        detailed_parts.append(", ".join(harmony_bits) + ".")
    if include_structure and s.get("sections"):
        detailed_parts.append(_structure_sentence(s))
    detailed = " ".join(detailed_parts)

    return {"short": short, "standard": standard, "detailed": detailed}


def _structure_sentence(s: dict[str, Any]) -> str:
    secs = s["sections"]
    bits = [f"{sec['role']} ({sec['duration_sec']:.0f}s)" for sec in secs]
    if len(bits) > 10:
        bits = bits[:9] + [f"... {len(secs) - 9} more sections"]
    return "Arrangement: " + " \u2192 ".join(bits) + f". Form {s.get('form', '')}."


# --------------------------------------------------------------------------
# Optional LLM refinement
# --------------------------------------------------------------------------

REFINE_SYSTEM = """You write prompts for text-to-music models.

You are given measured acoustic analysis of a track the user produced themselves.
Rewrite it as a single natural-language prompt a music generation model would
follow well.

Rules:
- Describe style, instrumentation, mood, rhythm feel, and production character.
- Keep every concrete measurement that matters: BPM, key, and arrangement shape.
- Do not invent instruments the analysis does not support. If the analysis is
  ambiguous about an instrument, describe the timbre instead of naming it.
- Do not reference any artist, band, song or album by name.
- One paragraph, no preamble, no bullet points, no quotes around the output.
"""


def refine_prompt(analysis: dict[str, Any], draft: str,
                  call_llm) -> str:
    """Improve a draft prompt with an LLM.

    `call_llm(system: str, user: str) -> str` is supplied by the caller, so this
    works with a local model, an API, or anything else.
    """
    payload = {
        "measured_analysis": _slim(analysis),
        "rule_based_draft": draft,
    }
    user = ("Here is the analysis and a mechanical draft. Produce the improved "
            "prompt.\n\n" + json.dumps(payload, indent=2))
    return call_llm(REFINE_SYSTEM, user).strip()


def _slim(a: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky arrays before sending to an LLM."""
    out = json.loads(json.dumps(a))
    h = out.get("harmony", {})
    h.pop("chroma_profile", None)
    if "chord_sequence" in h:
        h["chord_sequence"] = h["chord_sequence"][:16]
    return out
