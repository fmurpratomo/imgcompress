"""Command line front end."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from .core import INPUT_SUFFIXES, FileResult, compress_file
from .options import DEFAULT_PRESET, LOW_SLACK, PRESETS, Options


def parse_size(text: str) -> int:
    """Parse a byte budget: "50kb", "50 KB", "500k", "1mb", "51200", or "off"."""
    t = text.strip().lower().replace(" ", "")
    if t in ("off", "no", "none", "0"):
        return 0
    mult = 1
    for suffix, factor in (("kib", 1024), ("mib", 1024**2),
                           ("kb", 1024), ("mb", 1024**2),
                           ("k", 1024), ("m", 1024**2), ("b", 1)):
        if t.endswith(suffix):
            t, mult = t[: -len(suffix)], factor
            break
    try:
        value = float(t)
    except ValueError:
        raise SystemExit(f"invalid size {text!r}; try 50kb, 500k or 51200")
    n = int(value * mult)
    if n < 1024:
        raise SystemExit(f"size {text!r} is too small to hold an image")
    return n


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def collect_inputs(paths: List[str], recursive: bool) -> Tuple[List[Path], Optional[Path]]:
    """Expand the CLI arguments into a file list, plus a common root for mirroring."""
    files: List[Path] = []
    roots: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found = sorted(
                f for f in it if f.is_file() and f.suffix.lower() in INPUT_SUFFIXES
            )
            files.extend(found)
            roots.append(p)
        elif p.is_file():
            files.append(p)
            roots.append(p.parent)
        else:
            print(f"warning: no such file or directory: {p}", file=sys.stderr)

    # Deduplicate while keeping order.
    seen = set()
    unique = []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)

    root = roots[0] if len(set(roots)) == 1 else None
    return unique, root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imgcompress",
        description=(
            "Compress PNG and JPEG images to the smallest size that still meets a "
            "measured quality target (SSIM), rather than a guessed quality number. "
            "Every output also stays under a byte ceiling (50 KB by default), "
            "downscaling when the quality target alone cannot get there."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  imgcompress photo.jpg                     # writes photo_compressed.jpg, <= 50 KB\n"
            "  imgcompress ./pics -r -o ./small          # whole tree into ./small, all <= 50 KB\n"
            "  imgcompress ./pics -r --max-bytes 200kb   # a roomier ceiling\n"
            "  imgcompress ./pics -r --max-bytes off     # quality target only, no ceiling\n"
            "  imgcompress *.png -p small                # lean harder on size\n"
            "  imgcompress shot.png --in-place --overwrite\n"
            "  imgcompress photo.jpg --max-side 1920 -p balanced\n"
            "  imgcompress photos.png -f auto            # let it switch PNG->JPEG\n"
        ),
    )
    p.add_argument("inputs", nargs="+", help="image files and/or directories")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--out-dir", type=Path, help="write results here, mirroring the input tree")
    out.add_argument("--in-place", action="store_true", help="replace the source files")
    out.add_argument("--suffix", default="_compressed", help="suffix when writing next to the source (default: %(default)s)")
    out.add_argument("--overwrite", action="store_true", help="allow overwriting existing files")
    out.add_argument("-r", "--recursive", action="store_true", help="descend into subdirectories")
    out.add_argument("-f", "--format", dest="out_format", default="keep",
                     choices=["keep", "auto", "jpeg", "png", "webp"],
                     help="output format; 'auto' picks whichever of PNG/JPEG is smaller "
                          "at the same measured quality (default: keep)")
    out.add_argument("--max-bytes", default="50kb", metavar="SIZE",
                     help="hard ceiling on every output file; resolution is reduced, "
                          "aspect ratio intact, when the quality target alone cannot "
                          "reach it (default: %(default)s; 'off' for no ceiling)")
    out.add_argument("-n", "--dry-run", action="store_true", help="report savings without writing")

    qual = p.add_argument_group("quality")
    qual.add_argument("-p", "--preset", default=DEFAULT_PRESET, choices=list(PRESETS),
                      help="quality preset (default: %(default)s)")
    qual.add_argument("-t", "--target", type=float, metavar="SSIM",
                      help="explicit mean-SSIM target in 0..1, overrides --preset")
    qual.add_argument("--target-low", type=float, metavar="SSIM",
                      help=f"floor for the worst 1%% of the image "
                           f"(default: {LOW_SLACK:g}x the mean error budget)")
    qual.add_argument("--target-flat", type=float, metavar="LEVELS",
                      help="ceiling on banding in smooth areas, in 8-bit levels "
                           "(default: 100x the mean error budget)")
    qual.add_argument("--min-quality", type=int, default=30, help="lowest JPEG/WebP quality to consider")
    qual.add_argument("--max-quality", type=int, default=95, help="highest JPEG/WebP quality to consider")
    qual.add_argument("--lossless", action="store_true", help="PNG: never quantise colours")
    qual.add_argument("--max-side", type=int, metavar="PX", help="downscale so the long edge is at most PX")
    qual.add_argument("--metric-max-side", type=int, default=4096, metavar="PX",
                      help="cap the resolution quality is measured at; lowering it is "
                           "faster but flatters the result (default: %(default)s)")

    meta = p.add_argument_group("metadata")
    meta.add_argument("--keep-exif", action="store_true", help="preserve EXIF (dates, camera, GPS)")
    meta.add_argument("--strip-icc", action="store_true", help="drop the colour profile too")
    meta.add_argument("--background", default="255,255,255", metavar="R,G,B",
                      help="fill colour when transparency must be flattened for JPEG")

    misc = p.add_argument_group("misc")
    misc.add_argument("-j", "--workers", type=int, default=0, help="parallel workers (default: auto)")
    misc.add_argument("--json", action="store_true", help="emit machine-readable results")
    misc.add_argument("-q", "--quiet", action="store_true", help="only print the summary")
    return p


def options_from_args(a: argparse.Namespace) -> Options:
    try:
        bg = tuple(int(x) for x in a.background.split(","))
        if len(bg) != 3 or not all(0 <= v <= 255 for v in bg):
            raise ValueError
    except ValueError:
        raise SystemExit(f"invalid --background {a.background!r}; expected R,G,B in 0..255")

    target = a.target if a.target is not None else PRESETS[a.preset]
    if not 0.0 < target <= 1.0:
        raise SystemExit("--target must be between 0 and 1")
    if not 1 <= a.min_quality <= a.max_quality <= 100:
        raise SystemExit("--min-quality/--max-quality must satisfy 1 <= min <= max <= 100")

    return Options(
        target_ssim=target,
        target_ssim_low=a.target_low,
        target_flat_err=a.target_flat,
        metric_max_side=a.metric_max_side,
        min_quality=a.min_quality,
        max_quality=a.max_quality,
        png_lossless=a.lossless,
        out_format=a.out_format,
        max_side=a.max_side,
        background=bg,  # type: ignore[arg-type]
        keep_exif=a.keep_exif,
        keep_icc=not a.strip_icc,
    )


def format_row(res: FileResult, width: int) -> str:
    name = res.source.name
    if len(name) > width:
        name = name[: width - 1] + "…"
    if not res.ok:
        return f"  {name:<{width}}  !  {res.error}"

    e = res.encoded
    assert e is not None
    bar = "✓" if e.met_target else "!"
    detail = f"{e.fmt} {e.label}"
    if e.capped:
        # The quality target was given up to meet the ceiling, so the two
        # guards that go with it (worst-1%, banding) are not what decided this.
        detail += f", ssim {e.ssim:.4f}, size-limited"
    elif e.label != "original kept":
        detail += f", ssim {e.ssim:.4f}/{e.ssim_low:.3f}, band {e.flat_err:.2f}"
    if e.note:
        detail += f" [{e.note}]"
    return (
        f"  {name:<{width}}  {bar} {human(e.original_size):>10} -> {human(e.size):>10}"
        f"  {e.saved_pct:5.1f}%   {detail}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    opts = options_from_args(args)

    if args.in_place and args.out_dir:
        raise SystemExit("--in-place and --out-dir are mutually exclusive")
    if args.in_place and not args.overwrite and not args.dry_run:
        raise SystemExit("--in-place rewrites your originals; pass --overwrite to confirm")

    max_bytes = parse_size(args.max_bytes)
    files, root = collect_inputs(args.inputs, args.recursive)
    if not files:
        print("no images found", file=sys.stderr)
        return 1

    job = dict(
        opts=opts,
        max_bytes=max_bytes,
        out_dir=args.out_dir,
        root=root,
        suffix=args.suffix,
        in_place=args.in_place,
        overwrite=args.overwrite or args.in_place,
        dry_run=args.dry_run,
    )

    width = min(42, max(len(f.name) for f in files))
    if not args.quiet and not args.json:
        head = "DRY RUN - nothing will be written" if args.dry_run else f"Compressing {len(files)} image(s)"
        print(f"{head}\n  targets: mean ssim >= {opts.target_ssim:.3f}, "
              f"worst-1% >= {opts.effective_low():.3f}, "
              f"banding <= {opts.effective_flat():.2f} levels")
        if max_bytes:
            print(f"  size limit: {human(max_bytes)} per file; anything that cannot "
                  f"reach it at full resolution is downscaled")
        print()

    workers = args.workers or min(8, len(files), os.cpu_count() or 1)
    results: List[FileResult] = []

    if workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compress_file, f, **job): f for f in files}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if not args.quiet and not args.json:
                    print(format_row(res, width))
    else:
        for f in files:
            res = compress_file(f, **job)
            results.append(res)
            if not args.quiet and not args.json:
                print(format_row(res, width))

    results.sort(key=lambda r: str(r.source))

    ok = [r for r in results if r.ok and r.encoded]
    failed = [r for r in results if not r.ok]
    total_in = sum(r.encoded.original_size for r in ok)
    total_out = sum(r.encoded.size for r in ok)
    missed = [r for r in ok if not r.encoded.met_target]

    if args.json:
        print(json.dumps({
            "files": [
                {
                    "source": str(r.source),
                    "output": str(r.output) if r.output else None,
                    "error": r.error or None,
                    "original_bytes": r.encoded.original_size if r.encoded else None,
                    "compressed_bytes": r.encoded.size if r.encoded else None,
                    "saved_pct": round(r.encoded.saved_pct, 2) if r.encoded else None,
                    "format": r.encoded.fmt if r.encoded else None,
                    "setting": r.encoded.label if r.encoded else None,
                    "ssim": round(r.encoded.ssim, 5) if r.encoded else None,
                    "ssim_low": round(r.encoded.ssim_low, 5) if r.encoded else None,
                    "flat_err": round(r.encoded.flat_err, 3) if r.encoded else None,
                    "met_target": r.encoded.met_target if r.encoded else None,
                    "dimensions": [r.encoded.width, r.encoded.height] if r.encoded else None,
                    "size_limited": r.encoded.capped if r.encoded else None,
                }
                for r in results
            ],
            "total_original_bytes": total_in,
            "total_compressed_bytes": total_out,
            "total_saved_pct": round(100.0 * (total_in - total_out) / total_in, 2) if total_in else 0.0,
            "failed": len(failed),
        }, indent=2))
        return 1 if failed else 0

    pct = 100.0 * (total_in - total_out) / total_in if total_in else 0.0
    print(f"\n{len(ok)} image(s): {human(total_in)} -> {human(total_out)}   saved {human(total_in - total_out)} ({pct:.1f}%)")
    if max_bytes and ok:
        capped = [r for r in ok if r.encoded.capped]
        biggest = max(r.encoded.size for r in ok)
        print(f"largest output {human(biggest)}, ceiling {human(max_bytes)}; "
              f"{len(capped)} of {len(ok)} downscaled to fit")
    if missed:
        print(f"{len(missed)} image(s) could not reach the target even at max quality; "
              f"the best available encode was kept.")
    for r in failed:
        print(f"failed: {r.source}: {r.error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
