"""Tunable knobs shared by every encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: Named presets, mapped to the mean-SSIM the encoders must hit.
#: The comments are measured results on 12 MP camera photos - see README.
PRESETS = {
    "max": 0.995,       # ~jpeg q90, ~30% off. Visually lossless.
    "high": 0.990,      # ~jpeg q80, ~50% off. Artifacts need pixel-peeping.
    "balanced": 0.985,  # ~jpeg q70, ~64% off. Still clean at 100%.
    "small": 0.978,     # ~jpeg q57, ~71% off. Slight softening up close.
    "tiny": 0.965,      # ~jpeg q42, ~76% off. Fine for thumbnails/previews.
}

DEFAULT_PRESET = "high"

#: How much worse the worst regions may be than the average, as a multiple of
#: the mean's error budget. Calibrated on real photos so that the mean target
#: and the percentile floor bind at roughly the same quality: fine texture
#: naturally scores low on SSIM, so a tighter factor would reject good encodes,
#: and a looser one would let flat-area blocking through.
LOW_SLACK = 5.0

#: Banding budget, in 8-bit levels, per unit of the mean's error budget.
#: At the default target this allows 1.0 level of local mean shift in smooth
#: regions. Measured on real images: good encodes land under 0.75, while
#: visibly banded output starts around 1.5.
FLAT_SCALE = 100.0


@dataclass
class Options:
    # --- quality target -------------------------------------------------
    target_ssim: float = PRESETS[DEFAULT_PRESET]
    #: Floor for the 1st-percentile SSIM, i.e. the worst regions of the image.
    #: Derived from ``target_ssim`` via LOW_SLACK when left as None.
    target_ssim_low: Optional[float] = None
    #: Ceiling on banding in smooth regions, in 8-bit levels. Derived from
    #: ``target_ssim`` via FLAT_SCALE when left as None.
    target_flat_err: Optional[float] = None
    #: Quality is measured at full resolution up to this long side. 4096 covers
    #: any normal camera photo; the cap only bounds time and memory on huge
    #: scans. Lowering it makes the search faster but flatters the result,
    #: because downscaling averages compression artifacts away.
    metric_max_side: int = 4096

    # --- encoder search space -------------------------------------------
    min_quality: int = 30
    max_quality: int = 95
    #: Palette sizes tried for lossy PNG, chosen to land on PNG bit-depth
    #: boundaries (2/4/8 bpp) where the real savings are.
    png_palette_ladder: Tuple[int, ...] = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    png_lossless: bool = False

    # --- output ---------------------------------------------------------
    #: "keep", "auto", "jpeg", "png" or "webp".
    out_format: str = "keep"
    max_side: Optional[int] = None
    #: Background used when an image with alpha has to become JPEG.
    background: Tuple[int, int, int] = (255, 255, 255)
    keep_exif: bool = False
    keep_icc: bool = True

    def effective_low(self) -> float:
        if self.target_ssim_low is not None:
            return self.target_ssim_low
        return max(0.0, 1.0 - LOW_SLACK * (1.0 - self.target_ssim))

    def effective_flat(self) -> float:
        if self.target_flat_err is not None:
            return self.target_flat_err
        return FLAT_SCALE * (1.0 - self.target_ssim)

    def accepts(self, mean: float, low: float, flat: float = 0.0) -> bool:
        return (
            mean >= self.target_ssim
            and low >= self.effective_low()
            and flat <= self.effective_flat()
        )
