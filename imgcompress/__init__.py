"""Quality-targeted PNG/JPEG compression.

Instead of guessing a quality number, these encoders search for the smallest
file that still matches the original within a measured SSIM budget.

    from imgcompress import Options, compress_image, compress_file

    out = compress_image(Path("photo.jpg").read_bytes(), Options())
    print(out.saved_pct, out.ssim)
"""

from .core import Encoded, FileResult, compress_file, compress_image
from .options import DEFAULT_PRESET, PRESETS, Options

__version__ = "1.0.0"

__all__ = [
    "Options",
    "PRESETS",
    "DEFAULT_PRESET",
    "Encoded",
    "FileResult",
    "compress_image",
    "compress_file",
]
