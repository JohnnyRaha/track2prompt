# track2prompt

Analyse a track you produced and turn the measured features into a prompt for a
text-to-music model, including ready-to-paste fields for Suno.

Built in Python with librosa and scikit-learn, designed and iterated with an AI
coding assistant (Claude) from first idea to working tool.

Intended for **your own material** or licensed stems. It describes style,
instrumentation, structure and production character — it is not a way to
reproduce someone else's recording, and a text prompt cannot do that anyway.

## Install

```bash
git clone https://github.com/JohnnyRaha/track2prompt.git
cd track2prompt
pip install -r requirements.txt
```

Quick check with a synthetic track whose answers are known (124 BPM, A minor,
intro-A-B-A-outro):

```bash
python make_test_audio.py
python -m track2prompt.cli test_track.wav --style "deep house"
```

That prints:

```text
test_track.wav  (62s)
----------------------------------------------------------------
  tempo       123.0 BPM   (straight, tightly quantised)
  key         C major  (confidence 0.11)
  brightness  dark  (centroid 1092 Hz)
  dynamics    dynamic, open master  (crest 23.3 dB)
  image       very wide
  form        A-B-A
----------------------------------------------------------------

PROMPT

deep house, upbeat at 123 BPM, in C major, straight, tightly quantised. busy
rhythmic activity, bass-heavy, dark tonal balance. dynamic, open master, large
reverberant space, sharp, punchy transients, very wide stereo image. chords
drawn from C, Am, F, Gm, Em, chord changes about every 1 bar. Arrangement:
verse / development (15s) → chorus / peak section (31s) → outro (16s). Form
A-B-A.
```

Tempo and form match the ground truth. The key comes back as C major, the
relative major of A minor, with a confidence of only 0.11 — see
[Known limits](#known-limits).

Add `--suno --instrumental` for Suno-formatted fields instead:

```text
STYLE FIELD  [167/1000 chars]

deep house, instrumental, steady driving, deep bass, punchy drums, tight programmed groove, dynamic open mix, spacious reverb, wide stereo, dark tone, 123 BPM, C major

LYRICS FIELD  [89 chars]

[Verse]
[Instrumental - 15s]

[Chorus]
[Instrumental - 31s]

[Outro]
[Instrumental - 16s]

NEGATIVE TAGS

harsh treble, overcompressed, distorted
```

## Use

```bash
python -m track2prompt.cli mytrack.wav --style "lo-fi house" --json analysis.json
```

Options:

| flag | effect |
|---|---|
| `--style` | seeds the prompt with a genre hint you supply |
| `--variant` | `short` / `standard` / `detailed` (default detailed) |
| `--stems` | directory of Demucs stems, analysed individually |
| `--no-structure` | skip segmentation, roughly 3× faster |
| `--json` | write the full feature dump |
| `--suno` | print Suno-formatted fields instead of a prose prompt |
| `--suno-model` | `v4` / `v4.5` / `v5` / `v5.5` — sets the field limits (default v5) |
| `--instrumental` / `--vocals` | tells Suno whether to add a voice |
| `--title` | override the suggested title |

### Suno output

```bash
python -m track2prompt.cli mytrack.wav --style "lo-fi house" --suno --instrumental
```

Suno splits its input across fields that behave differently, so the tool
writes each one separately:

- **Style field**: comma-separated tags, genre first because earlier tags carry
  more weight, with BPM and key as bare text at the end. Capped at 1,000
  characters on v4.5 and later, 200 on v4.
- **Lyrics field**: the detected arrangement as section metatags
  (`[Verse]`, `[Chorus]`, `[Instrumental - 30s]`). Structure belongs here,
  never in the style field.
- **Negative tags**: steers away from the opposite of what was measured.

### With stems

Stem separation makes everything downstream more accurate, because pitch
tracking and chord recognition both degrade badly on a full mix.

```bash
demucs -o stems mytrack.wav
python -m track2prompt.cli mytrack.wav --stems stems/htdemucs/mytrack
```

### As a library

```python
from track2prompt import load, analyse_rhythm, analyse_harmony, build_prompt
from track2prompt.cli import analyse_file

analysis = analyse_file("mytrack.wav")
prompts  = build_prompt(analysis, style_hint="ambient techno")
print(prompts["detailed"])
```

## What it measures

**Rhythm** — tempo, beat grid, onset density, swing ratio, pulse clarity, and
timing variability (which separates programmed from played).

**Harmony** — key and mode via Krumhansl-Kessler correlation, chord sequence by
template matching with median smoothing, harmonic rhythm.

**Structure** — self-similarity matrix over chroma and MFCC, Foote checkerboard
novelty for boundaries, agglomerative clustering so repeated sections share a
label, then adjacent same-label runs are merged.

**Production** — spectral centroid and rolloff, flatness, crest factor, dynamic
range, six-band energy split, mid/side ratio, an RT60 proxy from post-onset
decay slopes, and transient sharpness.

## Optional LLM pass

`refine_prompt()` takes your own `call_llm(system, user) -> str`, so it works
with a local model or an API. The rule-based draft is always produced first, so
the LLM is improving prose rather than inventing facts.

```python
from track2prompt import refine_prompt
better = refine_prompt(analysis, prompts["detailed"], call_llm=my_llm)
```

## Known limits

**Relative keys.** C major and A minor contain the same pitch classes, so
correlation alone cannot separate them. There is a bass-register tiebreak, but
when the bass gives equal weight to both tonics it stays ambiguous. The output
reports `key_alternative` and a low `key_confidence` in that case — trust the
confidence, and override with your own knowledge of the track.

**Chord quality.** Only major and minor triads are templated. Sevenths,
suspensions and extensions get mapped to the nearest triad, so a Gsus2 reads as
`Gm`. Add templates to `_chord_templates()` if you need more.

**Loudness.** `loudness_dbfs_rms` is an RMS proxy, not true LUFS. Install
`pyloudnorm` and swap it in if the number matters.

**Section roles.** `intro` / `chorus` / `breakdown` are inferred from level and
position, not from musical understanding. The boundaries and the repetition
pattern are the reliable parts; the names are a convenience.

## Next step: close the loop

The interesting version measures whether the prompt actually worked. Encode the
original with CLAP, generate from the prompt, encode the result, take the cosine
similarity. That gives you a score — and once you have a score you can perturb
the prompt, regenerate, and keep what improves it, rather than guessing.

Validate the metric against your own ears on a handful of tracks before you
trust it to optimise anything.

## Licence

MIT. See `LICENSE`.
