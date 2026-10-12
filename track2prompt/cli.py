"""Command line entry point: audio file -> analysis JSON + prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import load, analyse_rhythm, analyse_harmony, analyse_production
from .structure import analyse_structure
from .prompt import build_prompt
from .suno import build_suno, format_suno


def analyse_file(path: str, with_structure: bool = True) -> dict[str, Any]:
    """Run the full analysis pipeline on one audio file."""
    track = load(path)
    result: dict[str, Any] = {
        "source": {
            "file": track.name,
            "duration_sec": round(track.duration, 2),
            "sample_rate": track.sr,
            "channels": 2 if track.stereo is not None else 1,
        },
        "rhythm": analyse_rhythm(track),
        "harmony": analyse_harmony(track),
        "production": analyse_production(track),
    }
    if with_structure:
        result["structure"] = analyse_structure(track)
    return result


def analyse_stems(stem_dir: str) -> dict[str, Any]:
    """Analyse a directory of Demucs stems, one entry per stem."""
    out = {}
    for f in sorted(Path(stem_dir).glob("*.wav")):
        track = load(str(f))
        out[f.stem] = {
            "rhythm": analyse_rhythm(track),
            "harmony": analyse_harmony(track),
            "production": analyse_production(track),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="track2prompt",
        description="Analyse an audio track and generate a music-model prompt.")
    ap.add_argument("audio", help="path to an audio file you produced")
    ap.add_argument("--stems", help="optional directory of separated stems")
    ap.add_argument("--style", help="style hint to seed the prompt, e.g. 'lo-fi house'")
    ap.add_argument("--json", dest="json_out", help="write full analysis JSON here")
    ap.add_argument("--no-structure", action="store_true",
                    help="skip section segmentation (faster)")
    ap.add_argument("--variant", choices=["short", "standard", "detailed"],
                    default="detailed")
    ap.add_argument("--suno", action="store_true",
                    help="emit Suno-formatted fields instead of prose")
    ap.add_argument("--suno-model", default="v5",
                    choices=["v4", "v4.5", "v5", "v5.5"],
                    help="field limits follow the model you generate with")
    voc = ap.add_mutually_exclusive_group()
    voc.add_argument("--instrumental", action="store_true",
                     help="track has no vocals")
    voc.add_argument("--vocals", action="store_true",
                     help="track has vocals")
    ap.add_argument("--title", help="override the suggested title")
    args = ap.parse_args(argv)

    if not Path(args.audio).exists():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 1

    analysis = analyse_file(args.audio, with_structure=not args.no_structure)
    if args.stems:
        analysis["stems"] = analyse_stems(args.stems)

    prompts = build_prompt(analysis, style_hint=args.style,
                           include_structure=not args.no_structure)
    analysis["prompts"] = prompts

    instrumental = True if args.instrumental else False if args.vocals else None
    if args.suno:
        analysis["suno"] = build_suno(analysis, style_hint=args.style,
                                      model=args.suno_model,
                                      instrumental=instrumental,
                                      title=args.title)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(analysis, indent=2))

    r, h, p = analysis["rhythm"], analysis["harmony"], analysis["production"]
    print(f"\n{analysis['source']['file']}  "
          f"({analysis['source']['duration_sec']:.0f}s)")
    print("-" * 64)
    print(f"  tempo       {r['tempo_bpm']} BPM   ({r['feel']})")
    print(f"  key         {h['key']} {h['mode']}  "
          f"(confidence {h['key_confidence']:.2f})")
    print(f"  brightness  {p['brightness']}  "
          f"(centroid {p['spectral_centroid_hz']:.0f} Hz)")
    print(f"  dynamics    {p['compression']}  "
          f"(crest {p['crest_factor_db']:.1f} dB)")
    print(f"  image       {p['stereo_width']['label']}")
    if analysis.get("structure", {}).get("form"):
        print(f"  form        {analysis['structure']['form']}")
    print("-" * 64)
    if args.suno:
        print()
        print(format_suno(analysis["suno"]))
        print()
    else:
        print("\nPROMPT\n")
        print(prompts[args.variant])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
