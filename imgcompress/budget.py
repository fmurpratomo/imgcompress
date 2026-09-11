"""Fitting an image into a hard byte budget, aspect ratio intact.

A 12 MP photo cannot reach 50 KB by quality alone - at full resolution even
the lowest quality is an order of magnitude too big, and forcing it there just
produces a blocky mess. Something has to give, and it should be resolution.

That makes it a two-dimensional problem: every (resolution, quality) pair that
fits the budget is a candidate, and they trade off against each other. A large
blocky one and a small clean one can weigh the same. So rather than guessing a
rule, this tries a ladder of resolutions, finds the best quality that fits at
each, and then *measures* which result actually looks best - comparing them all
against the original at one common viewing size, which is the only way to
compare across resolutions fairly.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import List, Optional, Tuple

from PIL import Image

from .metrics import compare
from .options import Options

#: Bits per pixel to aim for when picking candidate resolutions. Spanning
#: "heavily compressed" to "generous" turns the byte budget into a spread of
#: sensible pixel counts, whatever the budget happens to be. The high end
#: matters more than it looks: lossless PNG runs at 8+ bpp, so without these
#: rungs a small-but-clean lossless copy is never even considered.
_BPP_LADDER = (0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.2, 3.0, 4.5, 6.5, 9.0, 13.0)

#: SSIM penalty per 8-bit level of banding, when ranking candidates. Mean SSIM
#: on its own is blind to dithering in smooth areas - it happily prefers a
#: large speckled image to a smaller clean one. At this weight the ~1.5 levels
#: where banding becomes visible costs about as much as the gap between two
#: neighbouring candidates, which is the point at which the ranking flips.
_BAND_PENALTY = 0.01

#: Candidates are judged after being brought to a common long side. Large
#: enough to expose blockiness, small enough that scoring stays fast.
_VIEW_SIDE = 1024

#: Never propose something too small to be useful as a picture.
_MIN_SIDE = 48


def _sizes_for(size: Tuple[int, int], budget: int) -> List[Tuple[int, int]]:
    """Candidate pixel dimensions, largest first, aspect ratio preserved."""
    w, h = size
    total = float(w * h)
    seen = set()
    out: List[Tuple[int, int]] = []
    scales = sorted(
        {min(1.0, math.sqrt((8.0 * budget / bpp) / total)) for bpp in _BPP_LADDER}
        | {1.0},
        reverse=True,
    )
    for s in scales:
        # Derive the short side from the long one so the ratio survives
        # rounding as closely as whole pixels allow.
        if w >= h:
            nw = max(1, round(w * s))
            nh = max(1, round(nw * h / w))
        else:
            nh = max(1, round(h * s))
            nw = max(1, round(nh * w / h))
        if (nw, nh) in seen:
            continue
        if s < 1.0 and min(nw, nh) < _MIN_SIDE:
            continue
        seen.add((nw, nh))
        out.append((nw, nh))
    return out


def _largest_fitting(encode, ladder, budget: int):
    """Binary-search ``ladder`` (worst to best) for the best output that fits."""
    lo, hi = 0, len(ladder) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = encode(ladder[mid])
        if len(data) <= budget:
            best = (ladder[mid], data)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def fit_to_budget(
    ref: Image.Image,
    opts: Options,
    budget: int,
    fmt: str,
    extra: dict,
) -> Optional[Tuple[bytes, str, int, int, float, float, float]]:
    """Best-looking encode of ``ref`` that fits ``budget`` bytes.

    Returns ``(data, label, width, height, mean_ssim, flat_err, score)``, or
    None if even the smallest candidate overflows the budget. ``score`` is the
    banding-penalised ranking value and is what callers must compare across
    formats - ranking on raw SSIM would quietly undo the penalty.

    Every candidate is swept - no early exit. The score is not unimodal in
    resolution: when the PNG ladder crosses from a 256-colour palette to
    lossless, quality jumps back *up* at a smaller size, and bailing out on a
    couple of declining scores would walk straight past the best answer.
    """
    from .core import _decode, _encode_jpeg, _encode_webp, _quantize, _save_png

    # One common viewing size for every candidate: a 600px version upscaled to
    # this will look soft, a 2000px one squeezed down will look blocky, and the
    # score says which trade the image at hand actually prefers.
    scale = min(max(ref.size), _VIEW_SIDE) / float(max(ref.size))
    view = (
        ref.resize(
            (max(1, round(ref.width * scale)), max(1, round(ref.height * scale))),
            Image.LANCZOS,
        )
        if scale < 1.0
        else ref
    )

    qualities = list(range(opts.min_quality, opts.max_quality + 1))
    best = None  # (score, data, label, w, h)

    for (nw, nh) in _sizes_for(ref.size, budget):
        small = ref if (nw, nh) == ref.size else ref.resize((nw, nh), Image.LANCZOS)

        if fmt == "png":
            # None = no quantisation at all. Without that rung the ladder stops
            # at 256 colours, PNG's palette limit, and an image whose palette
            # version lands well under the ceiling has no way to spend what is
            # left - it would ship a dithered 11 KB file against a 50 KB budget.
            ladder = list(opts.png_palette_ladder) + [None]
            encode = lambda n, im=small: _save_png(
                im if n is None else _quantize(im, n, Image.FLOYDSTEINBERG), extra
            )
            describe = (
                lambda w, h, v: f"{w}x{h}, lossless"
                if v is None
                else f"{w}x{h}, {v}-colour palette"
            )
        elif fmt == "webp":
            ladder = qualities
            encode = lambda q, im=small: _encode_webp(im, q, extra)
            describe = lambda w, h, v: f"{w}x{h}, q{v}"
        else:
            ladder = qualities
            encode = lambda q, im=small: _encode_jpeg(im, q, 2, extra)
            describe = lambda w, h, v: f"{w}x{h} q{v} 4:2:0"

        found = _largest_fitting(encode, ladder, budget)
        if found is None:
            continue  # even the cheapest setting overflows at this resolution
        setting, data = found

        decoded = _decode(data)
        scored = (
            decoded.resize(view.size, Image.LANCZOS)
            if decoded.size != view.size
            else decoded
        )
        mean, _, flat = compare(view, scored, 0)
        score = mean - _BAND_PENALTY * flat
        if best is None or score > best[0]:
            best = (score, mean, flat, data, describe(nw, nh, setting), nw, nh)

    if best is None:
        return None
    score, mean, flat, data, label, nw, nh = best
    return data, label, nw, nh, mean, flat, score
