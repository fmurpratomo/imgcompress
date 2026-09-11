"""Perceptual quality metrics used to steer the encoders.

The compressors search for the smallest file whose decoded pixels still look
like the original. That search needs numbers, and file size alone won't do.

Three come out of `compare`, and an encode has to satisfy all three:

    mean      overall SSIM fidelity
    low       1st percentile of SSIM - how bad the *worst* regions get
    flat_err  banding in regions the original left smooth, in 8-bit levels

`low` catches localised damage that a good average would hide: a photo can
average 0.99 while one flat patch of sky shows obvious 8x8 blocking.

`flat_err` covers SSIM's blind spot. SSIM normalises by local contrast and
carries a large stabiliser (C2 = 58.5), so tiny variances in smooth areas
barely move it - a gradient quantised into visible bands still scores 0.991.
That is precisely where the eye is most sensitive. Measuring the local mean
shift, but only where the original was flat, separates the two cases cleanly:
on real images a good encode lands under 0.75 levels, while visible banding
starts around 1.5.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# Standard SSIM stabilisers for 8-bit data (Wang et al. 2004).
_C1 = (0.01 * 255.0) ** 2
_C2 = (0.03 * 255.0) ** 2

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

#: A region counts as "flat" when its local standard deviation is below this,
#: in 8-bit levels. Loose enough to include gentle gradients, which is where
#: banding shows up, tight enough to exclude real texture.
_FLAT_SIGMA = 2.0

#: Mean shifts are accumulated into a histogram so the percentile can be taken
#: across bands without keeping a full-size array per band.
_HIST_BINS = 2048
_HIST_MAX = 32.0


def _gaussian_kernel(size: int, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def _blur(a: np.ndarray, k: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Separable Gaussian blur, done in chunks so big images stay cheap.

    sliding_window_view materialises a (rows, cols, len(k)) view; multiplying
    the whole image at once would need len(k)x its memory, hence the chunking.
    """
    r = len(k) // 2

    pad = np.pad(a, ((0, 0), (r, r)), mode="symmetric")
    mid = np.empty_like(a)
    for i in range(0, a.shape[0], chunk):
        win = sliding_window_view(pad[i : i + chunk], len(k), axis=1)
        mid[i : i + chunk] = win @ k

    pad = np.pad(mid, ((r, r), (0, 0)), mode="symmetric")
    out = np.empty_like(a)
    for j in range(0, a.shape[1], chunk):
        win = sliding_window_view(pad[:, j : j + chunk], len(k), axis=0)
        out[:, j : j + chunk] = win @ k
    return out


def _ssim_slab(
    ref: np.ndarray, test: np.ndarray, k: np.ndarray, hist: Optional[np.ndarray] = None
) -> np.ndarray:
    """SSIM for one slab, optionally accumulating flat-region mean shifts.

    Intermediates are reused in place rather than kept as named arrays; a
    naive version holds ~12 full-size float planes at once, which is real
    money once you are measuring a 12 MP photo in eight parallel workers.
    """
    mu_r = _blur(ref, k)
    mu_t = _blur(test, k)

    # Needed for the flat-region guard, before mu_r/mu_t get squared below.
    shift = np.abs(mu_r - mu_t) if hist is not None else None

    mu_rt = mu_r * mu_t
    mu_r *= mu_r  # now mu_r^2
    mu_t *= mu_t  # now mu_t^2

    cov = _blur(ref * test, k)
    cov -= mu_rt
    var_r = _blur(ref * ref, k)
    var_r -= mu_r
    var_t = _blur(test * test, k)
    var_t -= mu_t

    if hist is not None:
        # var_r is the reference's local variance, so this picks the pixels
        # the original left smooth - the only place banding is visible.
        flat = var_r < (_FLAT_SIGMA * _FLAT_SIGMA)
        if flat.any():
            hist += np.histogram(
                shift[flat], bins=_HIST_BINS, range=(0.0, _HIST_MAX)
            )[0]
    del shift

    den = var_r
    den += var_t
    den += _C2
    del var_t
    mu_r += mu_t  # mu_r^2 + mu_t^2
    mu_r += _C1
    den *= mu_r

    num = cov
    num *= 2.0
    num += _C2
    mu_rt *= 2.0
    mu_rt += _C1
    num *= mu_rt

    num /= den
    return num


def ssim_map(ref: np.ndarray, test: np.ndarray, band: int = 512) -> np.ndarray:
    """Per-pixel SSIM between two float32 single-channel planes."""
    return _ssim_banded(ref, test, band)[0]


def _ssim_banded(
    ref: np.ndarray, test: np.ndarray, band: int = 512, want_flat: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    size = min(11, ref.shape[0], ref.shape[1])
    if size < 3:
        # Too small for a meaningful window; fall back to a global comparison.
        diff = float(np.mean(np.abs(ref - test)))
        return np.full((1, 1), max(0.0, 1.0 - diff / 255.0), dtype=np.float32), None
    if size % 2 == 0:
        size -= 1
    k = _gaussian_kernel(size)
    hist = np.zeros(_HIST_BINS, dtype=np.int64) if want_flat else None

    h = ref.shape[0]
    if h <= band:
        return _ssim_slab(ref, test, k, hist), hist

    # Process in horizontal bands with a halo, so interior rows see their real
    # neighbours instead of a reflected edge.
    halo = size // 2
    out = np.empty_like(ref)
    for y0 in range(0, h, band):
        y1 = min(h, y0 + band)
        a0 = max(0, y0 - halo)
        a1 = min(h, y1 + halo)
        # Halo rows are computed for context but must not be counted twice, so
        # each band accumulates into its own histogram and only its own rows
        # are folded back in.
        sub = np.zeros(_HIST_BINS, dtype=np.int64) if want_flat else None
        slab = _ssim_slab(ref[a0:a1], test[a0:a1], k, sub)
        keep = slice(y0 - a0, y0 - a0 + (y1 - y0))
        out[y0:y1] = slab[keep]
        if want_flat:
            hist += sub
    return out, hist


def _percentile_from_hist(hist: np.ndarray, q: float) -> float:
    total = int(hist.sum())
    if total == 0:
        return 0.0  # no flat regions at all: nothing for banding to show up in
    target = q / 100.0 * total
    idx = int(np.searchsorted(np.cumsum(hist), target))
    idx = min(idx, _HIST_BINS - 1)
    return (idx + 0.5) * (_HIST_MAX / _HIST_BINS)


def _luma(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) @ _LUMA


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info


def _flatten(img: Image.Image, gray: int = 128) -> Image.Image:
    """Composite onto neutral grey so alpha differences show up in the luma."""
    rgba = img.convert("RGBA")
    back = Image.new("RGBA", rgba.size, (gray, gray, gray, 255))
    return Image.alpha_composite(back, rgba).convert("RGB")


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    longest = max(img.size)
    if max_side <= 0 or longest <= max_side:
        return img
    scale = max_side / float(longest)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    # BOX = plain area averaging: it preserves artifact energy instead of
    # smoothing it away the way Lanczos would.
    return img.resize(size, Image.BOX)


def compare(
    ref: Image.Image, test: Image.Image, max_side: int = 4096
) -> Tuple[float, float, float]:
    """Return ``(mean_ssim, low_ssim, flat_err)`` for ``test`` against ``ref``.

    ``low_ssim`` is the 1st percentile of the SSIM map. ``flat_err`` is the
    99th percentile of the local mean shift over regions the reference left
    smooth, in 8-bit levels - higher means more visible banding.
    """
    if ref.size != test.size:
        raise ValueError(f"size mismatch: {ref.size} vs {test.size}")

    ref = _downscale(ref, max_side)
    test = _downscale(test, max_side)

    alpha = _has_alpha(ref) or _has_alpha(test)
    r_rgb = _flatten(ref) if alpha else ref
    t_rgb = _flatten(test) if alpha else test

    planes = [(_luma(r_rgb), _luma(t_rgb))]
    if alpha:
        planes.append(
            (
                np.asarray(ref.convert("RGBA").getchannel("A"), dtype=np.float32),
                np.asarray(test.convert("RGBA").getchannel("A"), dtype=np.float32),
            )
        )

    means, lows, flats = [], [], []
    for a, b in planes:
        m, hist = _ssim_banded(a, b, want_flat=True)
        means.append(float(np.mean(m)))
        lows.append(float(np.percentile(m, 1.0)))
        flats.append(_percentile_from_hist(hist, 99.0) if hist is not None else 0.0)

    # Worst channel wins: a perfect luma doesn't excuse a mangled alpha edge.
    return min(means), min(lows), max(flats)
