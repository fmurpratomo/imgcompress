# imgcompress

Compresses PNG and JPEG images to the **smallest file that still meets a measured
quality target** — instead of guessing at a quality number and hoping.

Most compressors ask you to pick "quality 80" and apply it to everything. But
quality 80 is wasteful on a flat screenshot and too harsh on a detailed photo.
This tool inverts that: you state how good the result has to *look*, and it
searches for the cheapest encode that gets there.

For every image it encodes, decodes, and compares the result to the original
using **SSIM**, then binary-searches the encoder settings for the smallest file
that still clears the bar. Nothing that fails the bar is ever written, and the
output is never larger than the input.

## Install

```bash
pip install -r requirements.txt
```

Or install the `imgcompress` command:

```bash
pip install -e .
```

## Usage

```bash
python3 -m imgcompress photo.jpg                  # writes photo_compressed.jpg
python3 -m imgcompress ./pics -r -o ./small       # a whole tree, in parallel
python3 -m imgcompress ./pics -r -o ./small -f auto
python3 -m imgcompress shot.png --in-place --overwrite
python3 -m imgcompress photo.jpg --max-side 1920 -p balanced
python3 -m imgcompress ./pics -r --dry-run        # report savings, write nothing
```

```
Compressing 5 image(s)
  targets: mean ssim >= 0.990, worst-1% >= 0.950, banding <= 1.00 levels

  big_photo.png       ✓     5.0 MB ->     1.0 MB   79.0%   jpeg q83 4:2:0, ssim 0.9906/0.979, band 0.00
  graphic_banner.jpg  ✓     1.7 MB ->   852.3 KB   51.3%   jpeg q80 4:2:0, ssim 0.9900/0.958, band 0.74
  icon_alpha.png      ✓   217.7 KB ->   157.9 KB   27.5%   png lossless, ssim 1.0000/1.000, band 0.00
  photo_camera.jpg    ✓     1.4 MB ->   735.6 KB   48.3%   jpeg q81 4:2:0, ssim 0.9932/0.953, band 0.69
  screenshot.png      ✓   257.4 KB ->    68.4 KB   73.4%   png 32-colour palette, dithered, ssim 0.9998/0.996, band 0.01

5 image(s): 8.6 MB -> 2.8 MB   saved 5.7 MB (67.1%)
```

## Presets

Pick with `-p`. Measured on the five real images above (photos, a UI screenshot,
an icon with transparency), with `-f auto`:

| preset | 5 MP photo | 12 MP photo | 12 MP graphic | screenshot | icon+alpha | **total** |
|---|---|---|---|---|---|---|
| `max`      | 76% | 24% | 27% | 73% | 27% | **56%** |
| `high` ⟵ default | 79% | 48% | 51% | 73% | 27% | **67%** |
| `balanced` | 81% | 62% | 65% | 76% | 27% | **73%** |
| `small`    | 83% | 70% | 73% | 80% | 27% | **77%** |
| `tiny`     | 85% | 78% | 80% | 80% | 27% | **81%** |

What they land on in practice:

- **`max`** — visually lossless, roughly JPEG q90. For archival or print masters.
- **`high`** (default) — roughly q80. Artifacts need pixel-peeping to find.
- **`balanced`** — roughly q70. Still clean at 100% zoom. A good choice for the web.
- **`small`** — roughly q57. Slight softening of fine texture up close.
- **`tiny`** — roughly q42. Fine for thumbnails and previews, not for originals.

For a specific number instead of a preset, use `-t/--target` (e.g. `-t 0.993`).

## How the quality target works

Three numbers come out of the comparison, and an encode has to satisfy **all
three** before it is allowed to win:

| | what it measures | default at `-p high` | override |
|---|---|---|---|
| mean SSIM | overall fidelity | ≥ 0.990 | `-t` |
| worst 1% | how bad the worst regions get | ≥ 0.950 | `--target-low` |
| banding | local mean shift in smooth areas, in 8-bit levels | ≤ 1.00 | `--target-flat` |

**The worst-1% floor** catches localised damage that a good average hides. A
photo can average 0.99 while one flat patch of sky shows obvious 8×8 blocking.
It defaults to 5× the mean's error budget, calibrated on real photos so both
guards bind at about the same quality: fine texture naturally scores low on
SSIM, so a tighter factor would reject good encodes and a looser one would let
blocking through.

**The banding guard** covers a blind spot in SSIM itself. SSIM normalises by
local contrast and carries a large stabiliser (C2 = 58.5), so the tiny
variances in smooth regions barely move it — a gradient quantised to 32 colours
until it is *visibly striped* still scores 0.991 mean and 0.963 worst-1%, and
sails past both SSIM guards. That is exactly where the eye is most sensitive.
Measuring the local mean shift, but only over pixels the original left smooth,
separates the cases cleanly:

| | mean | worst 1% | banding |
|---|---|---|---|
| 12 MP photo at q81 (good) | 0.9932 | 0.953 | **0.70** |
| screenshot, 32 colours (good) | 0.9998 | 0.996 | **0.01** |
| gradient, 32 colours (visibly striped) | 0.9910 | 0.963 | **1.51** |
| 12 MP photo at q40 (bad) | 0.9817 | 0.854 | **1.91** |

Good encodes land under 0.75; visible banding starts around 1.5. With the guard
in place the same gradient gets 64 colours instead of 32 and comes out clean.
It costs nothing on photos — grain is not banding, and textured regions are
excluded from the measurement entirely.

**Resolution matters too.** Quality is measured at full resolution (up to
`--metric-max-side`, default 4096). Measuring a 12 MP photo downscaled to
2048px reported SSIM 0.988 where the true value was 0.983, and the worst-1%
figure went from a reassuring 0.956 to an honest 0.866 — downscaling averages
artifacts away. Lowering `--metric-max-side` is faster, but it flatters the
result.

## What it does per format

**JPEG** — binary-searches quality, always with optimised Huffman tables and a
progressive scan (smaller, and renders sooner over a network). Tries 4:2:0
chroma subsampling first and falls back to full 4:4:4 chroma when subsampling
can't hit the target — which is what saturated edges like red text need.
Grayscale images are encoded as grayscale.

**PNG** — reduces the mode losslessly first (drops a fully-opaque alpha channel,
demotes grayscale-in-RGB to grayscale, builds an exact palette when the image
really has ≤256 colours), then searches for the fewest palette colours that
still clear the quality bar. The ladder is chosen to land on PNG's bit-depth
boundaries, where the real savings are. Dithering is searched with and without,
keeping whichever is smaller — it helps gradients but costs bytes. Use
`--lossless` to skip quantisation entirely and get a bit-exact result.

**`-f auto`** picks whichever of PNG or JPEG is smaller at the same measured
quality. This is the big one for photographic PNGs: one 5 MB screenshot-of-a-photo
in the table above goes from **4.9% saved to 79%** because PNG is simply the wrong
container for it. Auto never converts an image with real transparency to JPEG.
In the default `keep` mode the tool says so in a note rather than switching
formats behind your back.

**WebP** is available with `-f webp` if you want it, but the default leaves your
formats alone.

## Notes

- **Metadata**: EXIF is dropped by default (it carries GPS and camera data, and
  costs bytes); orientation is baked into the pixels first, so nothing rotates.
  Pass `--keep-exif` to preserve it. The ICC colour profile is **kept** by
  default, so colours don't shift — `--strip-icc` drops it.
- **Safety**: existing files are never overwritten without `--overwrite`, writes
  go through a temp file and an atomic rename, and `--in-place` additionally
  requires `--overwrite`. Original modification times are preserved.
- **Failures don't stop a batch** — a corrupt file is reported and the rest keep
  going. The exit code is non-zero if anything failed.
- If an image genuinely can't reach the target even at maximum quality (heavy
  grain, pure noise), it's marked `!` and the best available encode is kept.
  The tool says so rather than pretending it succeeded.
- Forcing a format (`-f png` on a photo, say) can legitimately produce a
  *larger* file. That is allowed, because you asked for it, and the row is
  marked `larger than the input`. In `keep` and `auto` mode the original is
  returned instead.
- **Quality is measured against the file you hand it**, not against some
  original that no longer exists. Running the tool on its own output will
  shrink it again, because the previous result is now the reference. Compress
  from your masters, once — don't re-run it over a folder repeatedly.
- Animated GIF/APNG is not a target: only the first frame is kept, and that's
  called out in the output.

## Library use

```python
from pathlib import Path
from imgcompress import Options, compress_image

out = compress_image(Path("photo.jpg").read_bytes(), Options(target_ssim=0.985))
print(f"{out.saved_pct:.1f}% smaller, ssim {out.ssim:.4f} "
      f"(worst 1%: {out.ssim_low:.3f}, banding: {out.flat_err:.2f})")
Path("photo.min.jpg").write_bytes(out.data)
```

`--json` gives the same detail from the CLI for scripting.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

49 tests, no dependencies beyond Pillow and numpy.
