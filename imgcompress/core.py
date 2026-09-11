"""Quality-targeted image compression.

Every encoder here works the same way: encode, decode, measure the result
against the original with SSIM, and keep searching for the *smallest* file
that still clears the quality bar. Nothing is shipped that fails the bar, and
nothing is ever written larger than the input.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

from .metrics import compare
from .options import Options

# Formats we will read and re-encode.
INPUT_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".webp"}

_EXT = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}

#: Every suffix that already denotes a given format, so that rewriting
#: "photo.jpeg" in place replaces it instead of dropping a "photo.jpg" beside it.
_SUFFIXES = {
    "jpeg": {".jpg", ".jpeg", ".jpe"},
    "png": {".png"},
    "webp": {".webp"},
}

Image.MAX_IMAGE_PIXELS = 512_000_000  # guard against decompression bombs


@dataclass
class Candidate:
    data: bytes
    label: str
    mean: float
    low: float
    flat: float = 0.0
    setting: object = None

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class Encoded:
    """Outcome of compressing one image."""

    data: bytes
    fmt: str
    label: str
    ssim: float
    ssim_low: float
    flat_err: float
    original_size: int
    width: int
    height: int
    met_target: bool
    note: str = ""

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def saved(self) -> int:
        return self.original_size - self.size

    @property
    def saved_pct(self) -> float:
        if not self.original_size:
            return 0.0
        return 100.0 * self.saved / self.original_size


# --------------------------------------------------------------------------
# loading / normalising
# --------------------------------------------------------------------------


@contextmanager
def _encoder_buffer(nbytes: int):
    """Temporarily widen Pillow's encoder buffer (a module global)."""
    previous = ImageFile.MAXBLOCK
    ImageFile.MAXBLOCK = max(previous, nbytes)
    try:
        yield
    finally:
        ImageFile.MAXBLOCK = previous


def _save_to_buffer(img: Image.Image, fmt: str, min_buffer: int = 0, **kw) -> bytes:
    """Encode into memory, working around Pillow's undersized encoder buffer.

    With optimize/progressive, Pillow guesses an output buffer of W*H bytes.
    That is not enough for high-entropy images at high quality - progressive
    4:4:4 on fine texture or grain overflows it, and libjpeg then fails the
    save with "broken data stream" after printing to stderr. Callers that know
    they are in that territory pass ``min_buffer`` to avoid the failure; the
    retry below is the backstop for everything else.
    """
    buf = BytesIO()
    try:
        with _encoder_buffer(min_buffer):
            img.save(buf, fmt, **kw)
    except OSError:
        w, h = img.size
        with _encoder_buffer(4 * w * h + (1 << 20)):
            buf = BytesIO()
            img.save(buf, fmt, **kw)
    return buf.getvalue()


def _decode(data: bytes) -> Image.Image:
    img = Image.open(BytesIO(data))
    img.load()
    return img


def _is_grayscale(img: Image.Image) -> bool:
    if img.mode in ("L", "LA"):
        return True
    if img.mode not in ("RGB", "RGBA"):
        return False
    a = np.asarray(img)
    return bool(np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]))


def _is_opaque(img: Image.Image) -> bool:
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return True
    alpha = img.convert("RGBA").getchannel("A")
    return alpha.getextrema()[0] == 255


def _flatten_onto(img: Image.Image, bg: Tuple[int, int, int]) -> Image.Image:
    rgba = img.convert("RGBA")
    back = Image.new("RGBA", rgba.size, bg + (255,))
    return Image.alpha_composite(back, rgba).convert("RGB")


def prepare(img: Image.Image, opts: Options, for_jpeg: bool) -> Image.Image:
    """Normalise the source into the exact pixels we intend to ship.

    The metric compares against *this* image, not the file on disk, so any
    resizing or alpha flattening is a deliberate change rather than measured
    quality loss.
    """
    img = ImageOps.exif_transpose(img) or img

    if img.mode in ("I", "I;16", "I;16B", "F"):
        img = img.convert("L")
    elif img.mode == "CMYK":
        img = img.convert("RGB")
    elif img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif img.mode == "PA":
        img = img.convert("RGBA")

    if opts.max_side:
        longest = max(img.size)
        if longest > opts.max_side:
            scale = opts.max_side / float(longest)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(size, Image.LANCZOS)

    if for_jpeg:
        if img.mode in ("RGBA", "LA") or "transparency" in img.info:
            img = _flatten_onto(img, opts.background)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.mode == "RGB" and _is_grayscale(img):
            img = img.convert("L")
    else:
        if _is_opaque(img):
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode == "LA":
                img = img.convert("L")
        if img.mode == "RGB" and _is_grayscale(img):
            img = img.convert("L")
        elif img.mode == "RGBA" and _is_grayscale(img):
            img = img.convert("LA")

    return img


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------


def _search(
    ladder: Sequence,
    encode: Callable[[object], bytes],
    ref: Image.Image,
    opts: Options,
    label: Callable[[object], str],
) -> Tuple[Optional[Candidate], Optional[Candidate]]:
    """Binary-search ``ladder`` (ordered worst to best) for the cheapest pass.

    Returns ``(passing, best_effort)``. Quality rises monotonically with the
    ladder index, so the first passing index is also the smallest file.
    """
    lo, hi = 0, len(ladder) - 1
    passing: Optional[Candidate] = None
    best_effort: Optional[Candidate] = None

    while lo <= hi:
        mid = (lo + hi) // 2
        setting = ladder[mid]
        data = encode(setting)
        mean, low, flat = compare(ref, _decode(data), opts.metric_max_side)
        cand = Candidate(data, label(setting), mean, low, flat, setting)

        if best_effort is None or mean > best_effort.mean:
            best_effort = cand

        if opts.accepts(mean, low, flat):
            passing = cand
            hi = mid - 1
        else:
            lo = mid + 1

    return passing, best_effort


# --------------------------------------------------------------------------
# JPEG
# --------------------------------------------------------------------------

_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}


def _encode_jpeg(img: Image.Image, quality: int, subsampling: int, extra: dict) -> bytes:
    w, h = img.size
    return _save_to_buffer(
        img,
        "JPEG",
        # Pillow's own guess tops out at 2*W*H and still overflows on dense
        # 4:4:4 content, so ask for headroom rather than encode twice.
        min_buffer=2 * w * h + (1 << 16),
        quality=quality,
        optimize=True,      # optimal Huffman tables, a few % for free
        progressive=True,   # usually smaller still, and renders sooner on the web
        subsampling=subsampling,
        **extra,
    )


def _jpeg_candidates(ref: Image.Image, opts: Options, extra: dict) -> List[Candidate]:
    qs = list(range(opts.min_quality, opts.max_quality + 1))
    out: List[Candidate] = []

    # Chroma subsampling is free savings on most photos, but it wrecks
    # saturated edges (red text, UI screenshots). Try it first; fall back to
    # full chroma when 4:2:0 either fails outright or needs a silly quality.
    for ss in (2, 0):
        if ref.mode == "L" and ss != 0:
            continue  # no chroma planes to subsample
        passing, effort = _search(
            qs,
            lambda q, ss=ss: _encode_jpeg(ref, q, ss, extra),
            ref,
            opts,
            lambda q, ss=ss: f"q{q} {_SUBSAMPLING[ss]}",
        )
        if passing:
            out.append(passing)
            if int(passing.setting) < 90:
                break  # 4:2:0 landed comfortably; full chroma can't beat it
        elif effort:
            out.append(effort)
    return out


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------


def _save_png(img: Image.Image, extra: dict) -> bytes:
    return _save_to_buffer(img, "PNG", optimize=True, compress_level=9, **extra)


def _exact_palette(img: Image.Image) -> Optional[Image.Image]:
    """Palette-ise losslessly when the image really has <=256 colours."""
    if img.mode not in ("RGB", "L"):
        return None
    rgb = img.convert("RGB")
    a = np.asarray(rgb, dtype=np.uint32)
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    uniq = np.unique(packed)
    if uniq.size > 256:
        return None

    idx = np.searchsorted(uniq, packed).astype(np.uint8)
    out = Image.fromarray(idx, mode="P")
    palette = bytearray()
    for c in uniq.tolist():
        palette += bytes(((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF))
    palette += bytes(3 * (256 - uniq.size))
    out.putpalette(bytes(palette))
    return out


def _quantize(img: Image.Image, colors: int, dither: int) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        # Only FASTOCTREE understands alpha; it writes a tRNS chunk.
        return img.convert("RGBA").quantize(
            colors=colors, method=Image.FASTOCTREE, dither=dither
        )
    return img.convert("RGB").quantize(
        colors=colors, method=Image.MEDIANCUT, dither=dither
    )


def _png_candidates(ref: Image.Image, opts: Options, extra: dict) -> List[Candidate]:
    out: List[Candidate] = []

    # 1. Lossless: mode reduction (done in prepare) + max zlib, plus an exact
    #    palette when the image is flat enough to allow one.
    out.append(Candidate(_save_png(ref, extra), "lossless", 1.0, 1.0, 0.0))
    exact = _exact_palette(ref)
    if exact is not None:
        out.append(Candidate(_save_png(exact, extra), "lossless palette", 1.0, 1.0, 0.0))

    if opts.png_lossless:
        return out

    # 2. Lossy: fewest palette colours that still clears the quality bar.
    #    Dithering buys quality on gradients but costs bytes, so we search with
    #    it on, then re-test the winner without it and keep the smaller file.
    ladder = list(opts.png_palette_ladder)
    passing, _ = _search(
        ladder,
        lambda n: _save_png(_quantize(ref, n, Image.FLOYDSTEINBERG), extra),
        ref,
        opts,
        lambda n: f"{n}-colour palette, dithered",
    )
    if passing:
        out.append(passing)
        n = int(passing.setting)
        undithered = _save_png(_quantize(ref, n, Image.NONE), extra)
        if len(undithered) < passing.size:
            mean, low, flat = compare(ref, _decode(undithered), opts.metric_max_side)
            if opts.accepts(mean, low, flat):
                out.append(
                    Candidate(undithered, f"{n}-colour palette", mean, low, flat, n)
                )
    return out


# --------------------------------------------------------------------------
# WebP (optional output format)
# --------------------------------------------------------------------------


def _encode_webp(img: Image.Image, quality: int, extra: dict) -> bytes:
    return _save_to_buffer(img, "WEBP", quality=quality, method=6, **extra)


def _webp_candidates(ref: Image.Image, opts: Options, extra: dict) -> List[Candidate]:
    qs = list(range(opts.min_quality, min(opts.max_quality, 100) + 1))
    passing, effort = _search(
        qs,
        lambda q: _encode_webp(ref, q, extra),
        ref,
        opts,
        lambda q: f"q{q}",
    )
    out = [c for c in (passing or effort,) if c]
    out.append(
        Candidate(
            _save_to_buffer(ref, "WEBP", lossless=True, quality=100, method=6, **extra),
            "lossless", 1.0, 1.0, 0.0,
        )
    )
    return out


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def _viable_formats(src_format: Optional[str], img: Image.Image, opts: Options) -> List[str]:
    if opts.out_format == "auto":
        # JPEG is only a fair contender when there is no transparency to lose.
        return ["png", "jpeg"] if _is_opaque(img) else ["png"]
    if opts.out_format != "keep":
        return [opts.out_format]
    fmt = (src_format or "").lower()
    if fmt in ("jpeg", "mpo"):
        return ["jpeg"]
    if fmt == "webp":
        return ["webp"]
    return ["png"]


def _candidates_for(fmt: str, ref: Image.Image, opts: Options, extra: dict) -> List[Candidate]:
    if fmt == "jpeg":
        return _jpeg_candidates(ref, opts, extra)
    if fmt == "webp":
        return _webp_candidates(ref, opts, extra)
    return _png_candidates(ref, opts, extra)


def compress_image(data: bytes, opts: Options) -> Encoded:
    """Compress one image held in memory. Never returns more bytes than it got."""
    src = _decode(data)
    src_format = src.format
    frames = getattr(src, "n_frames", 1)

    base = ImageOps.exif_transpose(src) or src
    formats = _viable_formats(src_format, base, opts)

    metadata: dict = {}
    if opts.keep_icc and src.info.get("icc_profile"):
        metadata["icc_profile"] = src.info["icc_profile"]
    if opts.keep_exif and src.info.get("exif"):
        metadata["exif"] = src.info["exif"]

    # Each format gets its own reference image, because each implies different
    # legitimate changes (JPEG has to flatten alpha). Quality is always judged
    # against the pixels that format is actually trying to reproduce.
    pool: List[Tuple[str, Candidate, Image.Image]] = []
    for fmt in formats:
        ref = prepare(src, opts, for_jpeg=(fmt == "jpeg"))
        extra = dict(metadata)
        if fmt == "png":
            extra.pop("exif", None)  # Pillow writes PNG eXIf only for some modes
        for cand in _candidates_for(fmt, ref, opts, extra):
            pool.append((fmt, cand, ref))

    passing = [x for x in pool if opts.accepts(x[1].mean, x[1].low, x[1].flat)]
    if passing:
        fmt, best, ref = min(passing, key=lambda x: x[1].size)
        met = True
    else:
        fmt, best, ref = max(pool, key=lambda x: x[1].mean)
        met = False

    notes = []
    if frames > 1:
        notes.append(f"only the first of {frames} frames was kept")

    # A photographic PNG that resisted quantisation is really a JPEG in
    # disguise; say so rather than quietly returning a 3% saving.
    if (
        opts.out_format == "keep"
        and fmt == "png"
        and "lossless" in best.label
        and _is_opaque(ref)
        and best.size > 0.7 * len(data)
    ):
        notes.append("photographic PNG - try --format auto for a much smaller JPEG")

    shape_kept = opts.out_format in ("keep", "auto") and ref.size == base.size
    if best.size >= len(data) and shape_kept:
        # Re-encoding gained nothing; shipping the original is strictly better.
        return Encoded(
            data=data,
            fmt=_viable_formats(src_format, base, Options())[0]
            if opts.out_format == "auto"
            else fmt,
            label="original kept",
            ssim=1.0,
            ssim_low=1.0,
            flat_err=0.0,
            original_size=len(data),
            width=base.width,
            height=base.height,
            met_target=True,
            note="; ".join(notes) or "already well compressed",
        )

    if best.size > len(data):
        # Only reachable once the branch above has declined to keep the
        # original: the caller forced a format or a resize that costs bytes.
        notes.append("larger than the input")

    return Encoded(
        data=best.data,
        fmt=fmt,
        label=best.label,
        ssim=best.mean,
        ssim_low=best.low,
        flat_err=best.flat,
        original_size=len(data),
        width=ref.width,
        height=ref.height,
        met_target=met,
        note="; ".join(notes),
    )


@dataclass
class FileResult:
    source: Path
    output: Optional[Path]
    encoded: Optional[Encoded]
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == ""


def output_path_for(src: Path, encoded_fmt: str, out_dir: Optional[Path], root: Optional[Path], suffix: str, in_place: bool) -> Path:
    ext = _EXT[encoded_fmt]
    if in_place and src.suffix.lower() in _SUFFIXES[encoded_fmt]:
        return src
    stem = src.stem + ("" if (out_dir or in_place) else suffix)
    if out_dir:
        rel = src.relative_to(root) if root else Path(src.name)
        return out_dir / rel.parent / (stem + ext)
    return src.with_name(stem + ext)


def compress_file(
    src: Path,
    opts: Options,
    out_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    suffix: str = "_compressed",
    in_place: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> FileResult:
    try:
        data = src.read_bytes()
        encoded = compress_image(data, opts)
    except Exception as exc:  # noqa: BLE001 - report, never abort the batch
        return FileResult(src, None, None, f"{type(exc).__name__}: {exc}")

    dest = output_path_for(src, encoded.fmt, out_dir, root, suffix, in_place)
    if dry_run:
        return FileResult(src, dest, encoded)

    if dest != src and dest.exists() and not overwrite:
        return FileResult(src, dest, encoded, f"refusing to overwrite {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp~")
        tmp.write_bytes(encoded.data)
        os.replace(tmp, dest)
        # Keep the original's timestamps so sync tools stay sane.
        st = src.stat()
        os.utime(dest, (st.st_atime, st.st_mtime))
    except Exception as exc:  # noqa: BLE001
        return FileResult(src, dest, encoded, f"{type(exc).__name__}: {exc}")

    return FileResult(src, dest, encoded)
