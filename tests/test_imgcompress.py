"""Test suite. Run with:  python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgcompress import Options, compress_file, compress_image  # noqa: E402
from imgcompress.cli import collect_inputs, main  # noqa: E402
from imgcompress.core import (  # noqa: E402
    _exact_palette,
    _is_grayscale,
    _is_opaque,
    output_path_for,
    prepare,
)
from imgcompress.metrics import _gaussian_kernel, _ssim_slab, compare, ssim_map  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def photo_like(w: int = 640, h: int = 480, seed: int = 0) -> Image.Image:
    """Gradients, shapes and grain - compressible, but not trivially so.

    The grain is generated at half resolution and scaled up. Real sensor noise
    is spatially correlated by the demosaic and the lens; i.i.d. per-pixel
    noise is not a photo, it is a worst case no encoder can reproduce, and
    testing against it would just measure the fixture.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 120 + 100 * np.sin(xx / 90.0) + 30 * np.cos(yy / 55.0)
    g = 130 + 90 * np.cos((xx + yy) / 110.0)
    b = 140 + 80 * np.sin(yy / 70.0)
    a = np.stack([r, g, b], axis=-1)
    a[h // 4 : h // 2, w // 4 : w // 2] = [210, 60, 40]
    a[2 * h // 3 :, : w // 3] = [30, 90, 180]

    grain = rng.normal(0, 4.0, ((h + 1) // 2, (w + 1) // 2, 3)).astype(np.float32)
    grain = np.asarray(
        Image.fromarray(np.clip(grain + 128, 0, 255).astype(np.uint8), "RGB").resize(
            (w, h), Image.BILINEAR
        ),
        dtype=np.float32,
    ) - 128.0
    return Image.fromarray(np.clip(a + grain, 0, 255).astype(np.uint8), "RGB")


def pure_noise(w: int = 300, h: int = 300) -> Image.Image:
    """Incompressible by construction - used to test the give-up path."""
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def flat_ui(w: int = 400, h: int = 300) -> Image.Image:
    img = Image.new("RGB", (w, h), (28, 30, 34))
    px = img.load()
    for y in range(40, 120):
        for x in range(20, 380):
            px[x, y] = (240, 240, 245)
    for y in range(160, 200):
        for x in range(20, 200):
            px[x, y] = (60, 140, 220)
    return img


def with_alpha(w: int = 200, h: int = 200) -> Image.Image:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (w / 3) ** 2
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 220, 80, 60
    a[..., 3] = np.where(mask, 255, 0)
    a[: h // 4, :, 3] = 128  # a genuinely semi-transparent band
    return Image.fromarray(a, "RGBA")


def gradient(w: int = 600, h: int = 400) -> Image.Image:
    """A smooth sky-like ramp - the case palette quantisation bands."""
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
    a = np.stack([60 + 120 * yy / h, 90 + 110 * yy / h, 180 + 60 * yy / h], axis=-1)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def to_bytes(img: Image.Image, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


# --------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------


class TestMetric(unittest.TestCase):
    def test_identical_images_score_one(self):
        img = photo_like()
        mean, low, flat = compare(img, img.copy())
        self.assertAlmostEqual(mean, 1.0, places=5)
        self.assertAlmostEqual(low, 1.0, places=5)
        self.assertLess(flat, 0.05)

    def test_degradation_lowers_score(self):
        img = photo_like()
        scores = []
        for q in (20, 50, 90):
            enc = to_bytes(img, "JPEG", quality=q)
            scores.append(compare(img, Image.open(io.BytesIO(enc)))[0])
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertLess(scores[2], 1.0)

    def test_banding_matches_single_slab(self):
        rng = np.random.default_rng(1)
        a = (rng.random((1300, 700), dtype=np.float32) * 255).astype(np.float32)
        b = a + rng.normal(0, 5, a.shape).astype(np.float32)
        one = _ssim_slab(a.copy(), b.copy(), _gaussian_kernel(11))
        many = ssim_map(a.copy(), b.copy(), band=256)
        np.testing.assert_allclose(one, many, atol=1e-5)

    def test_percentile_catches_local_damage(self):
        """A small ruined region must show up in `low` even when `mean` is fine."""
        img = photo_like()
        broken = img.copy()
        broken.paste(Image.new("RGB", (60, 60), (0, 0, 0)), (30, 30))
        mean, low, _ = compare(img, broken)
        self.assertGreater(mean, 0.95)
        self.assertLess(low, 0.5)

    def test_alpha_difference_is_detected(self):
        img = with_alpha()
        broken = img.copy()
        broken.putalpha(Image.new("L", img.size, 255))
        mean = compare(img, broken)[0]
        self.assertLess(mean, 0.9)

    def test_tiny_images_do_not_crash(self):
        for size in ((1, 1), (2, 3), (5, 5), (12, 4)):
            a = Image.new("RGB", size, (10, 20, 30))
            mean, low, _ = compare(a, a.copy())
            self.assertGreaterEqual(mean, 0.99)
            self.assertGreaterEqual(low, 0.99)

    def test_flat_error_detects_banding_that_ssim_misses(self):
        """The whole point of the third guard: SSIM alone waves banding through."""
        grad = gradient()
        banded = grad.quantize(
            colors=32, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG
        ).convert("RGB")
        mean, low, flat = compare(grad, banded)
        self.assertGreater(mean, 0.99)   # SSIM says this is fine...
        self.assertGreater(low, 0.95)    # ...and so does the percentile...
        self.assertGreater(flat, 1.4)    # ...but it is visibly banded.

    def test_flat_error_falls_as_colours_rise(self):
        grad = gradient()
        errs = []
        for n in (16, 32, 64, 128):
            q = grad.quantize(
                colors=n, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG
            ).convert("RGB")
            errs.append(compare(grad, q)[2])
        self.assertEqual(errs, sorted(errs, reverse=True))

    def test_flat_error_ignores_textured_images(self):
        """Noise is not banding: a grainy photo must not trip the guard."""
        img = photo_like()
        enc = to_bytes(img, "JPEG", quality=90, optimize=True, progressive=True)
        self.assertLess(compare(img, Image.open(io.BytesIO(enc)))[2], 1.0)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compare(Image.new("RGB", (10, 10)), Image.new("RGB", (11, 10)))


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------


class TestOptions(unittest.TestCase):
    def test_derived_floor(self):
        self.assertAlmostEqual(Options(target_ssim=0.99).effective_low(), 0.95, places=6)

    def test_explicit_floor_wins(self):
        o = Options(target_ssim=0.99, target_ssim_low=0.80)
        self.assertEqual(o.effective_low(), 0.80)

    def test_derived_banding_budget(self):
        self.assertAlmostEqual(Options(target_ssim=0.99).effective_flat(), 1.0, places=6)
        self.assertEqual(Options(target_flat_err=0.25).effective_flat(), 0.25)

    def test_accepts_requires_all_three(self):
        o = Options(target_ssim=0.99)
        self.assertTrue(o.accepts(0.995, 0.96, 0.5))
        self.assertFalse(o.accepts(0.995, 0.90, 0.5))   # good mean, bad worst-case
        self.assertFalse(o.accepts(0.980, 0.99, 0.5))   # good worst-case, bad mean
        self.assertFalse(o.accepts(0.995, 0.96, 2.0))   # both fine, but banded


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_grayscale_detection(self):
        self.assertTrue(_is_grayscale(Image.new("RGB", (8, 8), (77, 77, 77))))
        self.assertFalse(_is_grayscale(Image.new("RGB", (8, 8), (77, 78, 77))))
        self.assertTrue(_is_grayscale(Image.new("L", (8, 8), 12)))

    def test_opacity_detection(self):
        self.assertTrue(_is_opaque(Image.new("RGB", (8, 8))))
        self.assertTrue(_is_opaque(Image.new("RGBA", (8, 8), (1, 2, 3, 255))))
        self.assertFalse(_is_opaque(with_alpha()))

    def test_exact_palette_roundtrips_losslessly(self):
        img = flat_ui()
        pal = _exact_palette(img)
        self.assertIsNotNone(pal)
        self.assertEqual(pal.mode, "P")
        np.testing.assert_array_equal(np.asarray(pal.convert("RGB")), np.asarray(img))

    def test_exact_palette_declines_rich_images(self):
        self.assertIsNone(_exact_palette(photo_like()))

    def test_prepare_applies_exif_orientation(self):
        img = Image.new("RGB", (40, 20), (5, 5, 200))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90
        data = to_bytes(img, "JPEG", exif=exif)
        out = prepare(Image.open(io.BytesIO(data)), Options(), for_jpeg=True)
        self.assertEqual(out.size, (20, 40))

    def test_prepare_flattens_alpha_for_jpeg(self):
        out = prepare(with_alpha(), Options(background=(255, 255, 255)), for_jpeg=True)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.getpixel((5, 195)), (255, 255, 255))  # corner was transparent

    def test_prepare_resizes_to_max_side(self):
        out = prepare(photo_like(800, 400), Options(max_side=200), for_jpeg=False)
        self.assertEqual(max(out.size), 200)
        self.assertEqual(out.size, (200, 100))

    def test_prepare_demotes_grayscale_rgb(self):
        img = Image.merge("RGB", [Image.new("L", (20, 20), 90)] * 3)
        self.assertEqual(prepare(img, Options(), for_jpeg=True).mode, "L")


# --------------------------------------------------------------------------
# compression
# --------------------------------------------------------------------------


class TestCompress(unittest.TestCase):
    def test_jpeg_meets_target_and_shrinks(self):
        src = to_bytes(photo_like(800, 600), "JPEG", quality=97)
        opts = Options()
        out = compress_image(src, opts)
        self.assertEqual(out.fmt, "jpeg")
        self.assertTrue(out.met_target)
        self.assertTrue(opts.accepts(out.ssim, out.ssim_low, out.flat_err))
        self.assertLess(out.size, out.original_size)

    def test_stricter_preset_gives_bigger_file(self):
        src = to_bytes(photo_like(600, 400), "JPEG", quality=97)
        small = compress_image(src, Options(target_ssim=0.97))
        best = compress_image(src, Options(target_ssim=0.995))
        self.assertLess(small.size, best.size)
        self.assertLessEqual(small.ssim, best.ssim)

    def test_png_quantisation_shrinks_flat_art(self):
        src = to_bytes(flat_ui())
        out = compress_image(src, Options())
        self.assertEqual(out.fmt, "png")
        self.assertTrue(out.met_target)
        self.assertLess(out.size, out.original_size)

    def test_gradient_is_not_left_banded(self):
        """Regression: the default preset used to ship a visibly banded ramp."""
        src = to_bytes(gradient())
        opts = Options()
        out = compress_image(src, opts)
        self.assertLessEqual(out.flat_err, opts.effective_flat())
        self.assertLess(out.flat_err, 1.4)
        self.assertLess(out.size, out.original_size)

    def test_looser_presets_tolerate_more_banding(self):
        src = to_bytes(gradient())
        strict = compress_image(src, Options(target_ssim=0.995))
        loose = compress_image(src, Options(target_ssim=0.965))
        self.assertLess(strict.flat_err, loose.flat_err)
        self.assertLess(loose.size, strict.size)

    def test_png_lossless_mode_is_exact(self):
        src = to_bytes(photo_like(300, 200))
        out = compress_image(src, Options(png_lossless=True))
        np.testing.assert_array_equal(
            np.asarray(Image.open(io.BytesIO(out.data)).convert("RGB")),
            np.asarray(Image.open(io.BytesIO(src)).convert("RGB")),
        )

    def test_alpha_survives_png_compression(self):
        src = to_bytes(with_alpha())
        out = compress_image(src, Options())
        result = Image.open(io.BytesIO(out.data)).convert("RGBA")
        self.assertEqual(result.getpixel((5, 195))[3], 0)     # still transparent
        self.assertEqual(result.getpixel((100, 100))[3], 255)  # still opaque

    def test_dense_444_encode_does_not_overflow_the_buffer(self):
        """Regression: progressive 4:4:4 on noise used to raise OSError."""
        from imgcompress.core import _encode_jpeg

        img = pure_noise(400, 400)
        for q in (80, 90, 95):
            data = _encode_jpeg(img, q, 0, {})
            self.assertEqual(Image.open(io.BytesIO(data)).size, (400, 400))

    def test_never_grows_the_file(self):
        """An already-tight JPEG must come back no larger than it went in."""
        src = to_bytes(photo_like(400, 300), "JPEG", quality=35, optimize=True)
        out = compress_image(src, Options(target_ssim=0.999))
        self.assertLessEqual(out.size, out.original_size)

    def test_auto_format_switches_photo_png_to_jpeg(self):
        src = to_bytes(photo_like(600, 450))
        keep = compress_image(src, Options())
        auto = compress_image(src, Options(out_format="auto"))
        self.assertEqual(auto.fmt, "jpeg")
        self.assertLess(auto.size, keep.size)

    def test_auto_format_keeps_png_when_alpha_present(self):
        out = compress_image(to_bytes(with_alpha()), Options(out_format="auto"))
        self.assertEqual(out.fmt, "png")

    def test_forced_jpeg_from_png_input(self):
        out = compress_image(to_bytes(flat_ui()), Options(out_format="jpeg"))
        self.assertEqual(out.fmt, "jpeg")
        self.assertEqual(Image.open(io.BytesIO(out.data)).format, "JPEG")

    def test_webp_output(self):
        out = compress_image(to_bytes(photo_like(300, 300)), Options(out_format="webp"))
        self.assertEqual(Image.open(io.BytesIO(out.data)).format, "WEBP")

    def test_resize_is_reflected_in_result(self):
        out = compress_image(to_bytes(photo_like(800, 600), "JPEG", quality=95),
                             Options(max_side=320))
        self.assertEqual((out.width, out.height), (320, 240))
        self.assertEqual(Image.open(io.BytesIO(out.data)).size, (320, 240))

    def test_exif_dropped_by_default_and_kept_on_request(self):
        img = photo_like(200, 150)
        exif = Image.Exif()
        exif[0x010F] = "TestCam"
        src = to_bytes(img, "JPEG", quality=97, exif=exif)
        self.assertNotIn(0x010F, Image.open(io.BytesIO(
            compress_image(src, Options()).data)).getexif())
        self.assertIn(0x010F, Image.open(io.BytesIO(
            compress_image(src, Options(keep_exif=True)).data)).getexif())

    def test_unreachable_target_is_reported_not_faked(self):
        """Pure noise cannot be compressed to a high SSIM; say so, don't pretend."""
        src = to_bytes(pure_noise(), "PNG")
        # Cap the search space so the target is genuinely out of reach.
        out = compress_image(
            src, Options(target_ssim=0.999, out_format="jpeg", max_quality=40)
        )
        self.assertFalse(out.met_target)
        self.assertLess(out.ssim, 0.999)
        Image.open(io.BytesIO(out.data)).load()  # still returns a usable image

    def test_corrupt_input_raises(self):
        with self.assertRaises(Exception):
            compress_image(b"this is not an image", Options())


# --------------------------------------------------------------------------
# files and CLI
# --------------------------------------------------------------------------


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name: str, img: Image.Image, **kw) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(to_bytes(img, "JPEG" if p.suffix == ".jpg" else "PNG", **kw))
        return p

    def test_output_path_variants(self):
        src = Path("/a/b/pic.png")
        self.assertEqual(output_path_for(src, "png", None, None, "_c", False),
                         Path("/a/b/pic_c.png"))
        self.assertEqual(output_path_for(src, "png", None, None, "_c", True), src)
        self.assertEqual(output_path_for(src, "jpeg", None, None, "_c", True),
                         Path("/a/b/pic.jpg"))
        # In place on a .jpeg must replace it, not drop a .jpg beside it.
        for name in ("pic.jpeg", "pic.JPG", "pic.jpe"):
            p = Path("/a/b") / name
            self.assertEqual(output_path_for(p, "jpeg", None, None, "_c", True), p)
        self.assertEqual(
            output_path_for(Path("/a/b/sub/pic.png"), "png", Path("/out"), Path("/a/b"), "_c", False),
            Path("/out/sub/pic.png"))

    def test_compress_file_writes_and_preserves_mtime(self):
        src = self.write("p.jpg", photo_like(400, 300), quality=97)
        before = src.stat().st_mtime
        res = compress_file(src, Options(), out_dir=self.dir / "out", root=self.dir)
        self.assertTrue(res.ok, res.error)
        self.assertTrue(res.output.exists())
        self.assertLess(res.output.stat().st_size, src.stat().st_size)
        self.assertAlmostEqual(res.output.stat().st_mtime, before, places=0)

    def test_dry_run_writes_nothing(self):
        src = self.write("p.png", flat_ui())
        res = compress_file(src, Options(), out_dir=self.dir / "out", dry_run=True)
        self.assertTrue(res.ok)
        self.assertFalse(res.output.exists())

    def test_refuses_to_clobber_without_permission(self):
        src = self.write("p.png", flat_ui())
        (self.dir / "p_compressed.png").write_bytes(b"keep me")
        res = compress_file(src, Options())
        self.assertFalse(res.ok)
        self.assertIn("refusing to overwrite", res.error)
        self.assertEqual((self.dir / "p_compressed.png").read_bytes(), b"keep me")

    def test_unreadable_file_is_reported_not_raised(self):
        bad = self.dir / "bad.png"
        bad.write_bytes(b"nope")
        res = compress_file(bad, Options())
        self.assertFalse(res.ok)
        self.assertIsNone(res.encoded)

    def test_collect_inputs_recursion_and_filtering(self):
        self.write("a.png", flat_ui())
        self.write("nested/b.jpg", photo_like(100, 100), quality=90)
        (self.dir / "notes.txt").write_text("ignore me")
        shallow, _ = collect_inputs([str(self.dir)], recursive=False)
        deep, root = collect_inputs([str(self.dir)], recursive=True)
        self.assertEqual([p.name for p in shallow], ["a.png"])
        self.assertEqual(sorted(p.name for p in deep), ["a.png", "b.jpg"])
        self.assertEqual(root, self.dir)

    def test_collect_inputs_deduplicates(self):
        p = self.write("a.png", flat_ui())
        files, _ = collect_inputs([str(p), str(p)], recursive=False)
        self.assertEqual(len(files), 1)

    def test_cli_end_to_end(self):
        self.write("a.png", flat_ui())
        self.write("b.jpg", photo_like(300, 200), quality=97)
        out = self.dir / "out"
        code = main([str(self.dir), "-o", str(out), "-q", "-j", "1"])
        self.assertEqual(code, 0)
        self.assertTrue((out / "a.png").exists())
        self.assertTrue((out / "b.jpg").exists())

    def test_cli_rejects_conflicting_flags(self):
        self.write("a.png", flat_ui())
        with self.assertRaises(SystemExit):
            main([str(self.dir), "--in-place", "-o", str(self.dir / "x")])
        with self.assertRaises(SystemExit):
            main([str(self.dir), "--in-place"])  # no --overwrite

    def test_in_place_replaces_jpeg_extension_file(self):
        src = self.dir / "p.jpeg"
        src.write_bytes(to_bytes(photo_like(300, 200), "JPEG", quality=97))
        before = src.stat().st_size
        self.assertEqual(main([str(src), "--in-place", "--overwrite", "-q", "-j", "1"]), 0)
        self.assertLess(src.stat().st_size, before)
        self.assertFalse((self.dir / "p.jpg").exists())

    def test_cli_in_place(self):
        src = self.write("a.png", flat_ui())
        before = src.stat().st_size
        self.assertEqual(main([str(src), "--in-place", "--overwrite", "-q", "-j", "1"]), 0)
        self.assertLess(src.stat().st_size, before)
        Image.open(src).load()  # still a valid image


if __name__ == "__main__":
    unittest.main(verbosity=2)
