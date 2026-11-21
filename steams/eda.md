# EDA observations: genre × stem mel-spectrogram grids

## How to read these figures

- Each figure shows **one chunk of 10.24 s per genre** (seed 0). Columns are the stems **bass, drums, vocals, other**.
- Axes: x = time in seconds, y = mel bin (0–127). Colors are AST-normalized log-mel values on **one shared scale** (about −1.25 = black/silence to about +0.95 = brightest).
- These are visual observations of **one chunk per genre**, so treat them as hypotheses to check with numbers, not as dataset statistics.

---

## 1. `flattened_clean.png`

The reference figure: clean stems, one file per stem.

| Genre | Bass | Drums | Vocals | Other |
|---|---|---|---|---|
| blues | Sparse. Low-mel notes with vertical spikes from about 3.3 s. Thin bright band at the top-left (about mel 125). | Loud, broadband, dense up to about 9.5 s. | Dark for about 6 s, then bright curved harmonic sweeps (6.5–8.5 s). | Dense horizontal harmonics with a dark gap around 7.5–9 s. |
| classical | Black (silent). | Black, apart from tiny specks near 6.5–7 s (mel 95–115). | Black (silent). | The only stem with content: dense horizontal harmonic lines for the whole chunk. |
| country | Dark with periodic vertical spikes, decaying low-mel notes, and one thick stripe near 1.5 s. | Regular vertical stripes with dark gaps. | Strong harmonic stacks, a gap around 7–8.5 s, then a return. | Bright and dense. |
| disco | Thin vertical lines and low-mel blocks. A block at 0–1.2 s around mel 80–90. | Very regular, dense vertical stripes. | **Unusual.** A uniform bright wash across all mel bins, brightest above mel ~110. It does not look like a voice. | Dense harmonics, dark at the top-right. |
| hiphop | Regular wedge-shaped bursts (kick/808-like) about 1 s apart. | Stripe pattern with dark gaps. | Dense harmonics with wavy, vibrato-like lines for the whole chunk. | Sparse bursts with dark gaps. |
| jazz | Stepped low-mel notes. | **Atypical.** Black bars at 1–3 s and 7.5–10 s and a large bright block at 3–8 s, instead of periodic stripes. | Nearly black (a few specks). | Dense harmonics, brighter segment around 7–9 s. |
| metal | Energy up to about mel 75 and black above. | Very dense, bright, broadband. | Dark until about 6.5 s, then strong energy to about 8.5 s. | Very bright in mid mels (about 40–100), which fits distorted guitars. Dark after about 8 s. |
| pop | Sparse vertical spikes and low-mel blobs. | Dense with dark patches. | Dense and bright. | Horizontal harmonic lines with a dark band at the top. |
| reggae | Low-mel notes with black gaps at 1.5–3 s and 8–9.5 s. | Dense. | Sparse bursts, strongest at about 7–9 s. | Blocky harmonic bursts. |
| rock | Continuous low-mel energy reaching about mel 80. | Regular stripes. | Strong harmonics for about 6 s, dark in the lower half at 6–8 s. | Dense harmonic texture. |

Points worth noting:

- **Stem order and alignment look right.** Drums show vertical transient stripes, bass sits in the low mels, vocals have stacked harmonic curves, and "other" is a dense harmonic carpet.
- **Silent stems:**
  - Classical has 3 of its 4 stems silent, which is expected for orchestral or piano recordings, but it means those samples carry only "black" input.
  - Jazz vocals are also silent. Silence is itself a class signal here (silent vocals and drums point to classical), but separated stems from Demucs will contain faint leakage instead of exact silence. That is a train/test gap to keep in mind.
- **Black strip on the right** (from about 9.5 s) in several panels is zero-padding of the last chunk of a 30 s song (3 × 10.24 s = 30.72 s).
- **Thin horizontal lines in the lowest mel bins** appear in almost every panel. This is typical of the very narrow low-frequency mel filters and is not a problem.

## 2. `flattened_augmented.png`

Same chunks as above, but with ESC-50 noise mixed in per stem.

Clearly visible changes compared with the clean figure:

- **country bass:** a large orange/purple block from about 3.5 to 9.5 s with horizontal harmonic lines, which looks like a sustained broadband environmental sound.
- **disco bass:** cloud-like blobs at about 5–9 s in the mid mels.
- **disco vocals:** a faint cloud pattern at about 3–8 s on top of the wash.
- **blues vocals:** striations and haze in the first 6 s, where the clean stem was nearly black.
- **metal vocals:** horizontal stripes at about 1.5–6.5 s in the previously dark region.
- **pop bass:** a hazy patch with arcs at about 4–7 s.
- **reggae bass:** broadband haze at about 1.5–6 s.
- **rock vocals:** the dark region near 5–6.5 s is changed.

The **drums** and **other** columns look almost identical to the clean version for all ten genres. Silent stems (classical, jazz vocals) are unchanged, which is by design.

## 3. `flattened_clean_vs_augmented.png`

Left half is clean, right half is augmented, for the same chunk and the same color scale.

- This is the best figure for judging the effect of the noise, since the two halves show exactly the same audio.
- Roughly **8 of the 40 stem panels** show an obvious change, almost all in **bass and vocals**. Drums and other change very little.
- With `noise_prob = 0.8`, about 30 of the non-silent stems should carry noise, so most of the added noise is **invisible at this scale**. A likely reason: the noise level is set relative to the stem's *average* power (SNR 8–18 dB), and on a log-mel scale a noise event 8–18 dB below a dense, loud stem is largely masked. Quiet or sparse stems (bass, vocals) show it clearly.
- This is not necessarily a bug, but it means the augmentation mostly teaches robustness on bass and vocals. It should be verified with numbers (see the next steps).
- Cosmetic: the "10" on the last tick of each panel collides with the "0" of the next one in the bottom row.

## 4. `grouped_augmented.png`

Each file holds all four stems, with noise applied per stem.

- Visible noise events (the exact events differ from the flattened version because the random draws are independent):
  - **blues bass:** a broad purple block at about 1.8–6.4 s reaching the top mel bins.
  - **country bass:** a broad block at about 1.5–6 s plus curved harmonic sweeps at about 6–8 s, like a tonal animal or siren-type sound.
  - **pop bass:** a block at about 3–7 s with horizontal lines.
  - **reggae bass:** a large orange block at about 2–7 s. **Reggae vocals:** a block at about 4–8 s in the lower half with arcs.
  - **rock bass:** a bright vertical stripe near 4.2 s, a short impulsive sound.
  - **metal drums:** a flat haze from about 3.5 s onward, a rare case of noise showing in a drums stem.
  - **hiphop bass:** a faint haze at about 1.5–3.5 s.
- As before, classical (and jazz vocals) are unaffected.

## 5. `grouped_clean.png`

- For **classical, country, hiphop, jazz, metal, pop, reggae and rock** these panels match `flattened_clean.png`.
- For **blues and disco** they show a **different chunk** from the flattened figures (for example, the blues bass and disco bass look different). The two figures use different pickers, so this is not an error, but it means flattened and grouped panels should not be compared directly for those two genres.
- Everything said in section 1 (silent stems, padding strip, disco vocals and jazz drums looking atypical) applies here too. The padding strip is visible in several stems (for example disco and classical).

## 6. `grouped_clean_vs_augmented.png`

- The same-chunk comparison for the grouped files. It confirms the pattern from section 3: the clearest changes are in **bass** (blues, country, pop, reggae, rock, hiphop) and **vocals** (blues, country, reggae). Drums and other are mostly unchanged. Metal drums and disco other show a lighter haze.
- Roughly **10–11 of the 40 panels** change visibly, slightly more than in the flattened pair.

---
