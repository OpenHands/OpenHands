"""Unit tests for Pillow (PIL) integration in OpenHands browser module.

These tests verify that OpenHands' image_to_png_base64_url and png_base64_url_to_image
functions work correctly with the updated pillow version.
"""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from openhands.runtime.browser.base64 import (
    image_to_png_base64_url,
    png_base64_url_to_image,
)


class TestImageToPngBase64Url:
    """Tests for the image_to_png_base64_url function."""

    def test_numpy_rgb_array_conversion(self):
        """Test conversion of RGB numpy array to base64 PNG."""
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel

        result = image_to_png_base64_url(img_array)

        assert isinstance(result, str)
        assert len(result) > 0

        # Verify round-trip
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (100, 100)
        assert img.mode == 'RGB'

    def test_numpy_rgba_array_conversion(self):
        """Test conversion of RGBA numpy array to base64 PNG (mode conversion)."""
        img_array = np.zeros((50, 50, 4), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red
        img_array[:, :, 3] = 128  # Half transparency

        result = image_to_png_base64_url(img_array)

        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (50, 50)
        # RGBA should be converted to RGB by the function
        assert img.mode == 'RGB'

    def test_pil_image_conversion(self):
        """Test conversion of PIL Image to base64 PNG."""
        img = Image.new('RGB', (200, 150), color=(0, 255, 0))

        result = image_to_png_base64_url(img)

        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.size == (200, 150)
        assert decoded_img.mode == 'RGB'

    def test_rgba_pil_image_conversion(self):
        """Test conversion of RGBA PIL Image to base64 PNG (mode conversion)."""
        img = Image.new('RGBA', (75, 75), color=(0, 0, 255, 128))

        result = image_to_png_base64_url(img)

        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.size == (75, 75)
        assert decoded_img.mode == 'RGB'

    def test_la_mode_conversion(self):
        """Test conversion of LA (grayscale with alpha) mode image."""
        img = Image.new('LA', (80, 80), color=(128, 200))

        result = image_to_png_base64_url(img)

        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.mode == 'RGB'

    def test_data_prefix_option(self):
        """Test the add_data_prefix option."""
        img = Image.new('RGB', (10, 10), color=(100, 100, 100))

        result_no_prefix = image_to_png_base64_url(img, add_data_prefix=False)
        assert not result_no_prefix.startswith('data:image/png;base64,')

        result_with_prefix = image_to_png_base64_url(img, add_data_prefix=True)
        assert result_with_prefix.startswith('data:image/png;base64,')

        # Base64 content should match
        assert result_with_prefix.split(',')[1] == result_no_prefix

    def test_various_image_sizes(self):
        """Test conversion with various image sizes."""
        sizes = [(1, 1), (10, 10), (100, 100), (640, 480), (1920, 1080)]

        for width, height in sizes:
            img = Image.new('RGB', (width, height), color=(50, 100, 150))
            result = image_to_png_base64_url(img)

            decoded = base64.b64decode(result)
            decoded_img = Image.open(io.BytesIO(decoded))
            assert decoded_img.size == (width, height)


class TestPngBase64UrlToImage:
    """Tests for the png_base64_url_to_image function."""

    def test_base64_without_prefix(self):
        """Test decoding base64 string without data URL prefix."""
        original = Image.new('RGB', (50, 50), color=(255, 0, 0))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode()

        result = png_base64_url_to_image(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

    def test_base64_with_data_prefix(self):
        """Test decoding base64 string with data URL prefix."""
        original = Image.new('RGB', (100, 75), color=(0, 255, 0))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_str = (
            'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        )

        result = png_base64_url_to_image(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 75)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion: image -> base64 -> image."""
        original = Image.new('RGB', (30, 30), color=(128, 64, 192))

        base64_str = image_to_png_base64_url(original)
        result = png_base64_url_to_image(base64_str)

        assert result.size == original.size
        assert result.mode == 'RGB'
        assert original.getpixel((15, 15)) == result.getpixel((15, 15))

    def test_roundtrip_with_drawing(self):
        """Test roundtrip with an image containing drawn content."""
        original = Image.new('RGB', (200, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(original)
        draw.rectangle([10, 10, 50, 50], fill=(255, 0, 0))
        draw.ellipse([60, 10, 100, 50], fill=(0, 0, 255))

        base64_str = image_to_png_base64_url(original)
        result = png_base64_url_to_image(base64_str)

        # Verify drawn content is preserved
        assert result.getpixel((30, 30)) == (255, 0, 0)  # Red rectangle
        assert result.getpixel((80, 30)) == (0, 0, 255)  # Blue ellipse
        assert result.getpixel((150, 50)) == (255, 255, 255)  # White background


class TestScreenshotSaveWorkflow:
    """Tests for the screenshot saving workflow used in utils.py browse function."""

    def test_screenshot_save_and_verify_workflow(self):
        """Test the screenshot save workflow: base64 decode -> save -> verify."""
        # Simulate a browser screenshot as base64 (this is what browse() receives)
        original = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(original)
        draw.rectangle([0, 0, 800, 50], fill=(60, 60, 60))  # Header bar
        draw.rectangle([100, 150, 200, 180], fill=(0, 120, 215))  # Button

        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_screenshot = base64.b64encode(buffer.getvalue()).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_path = Path(tmpdir) / 'screenshot.png'

            # Primary method: Direct base64 decode and save (from utils.py)
            image_data = base64.b64decode(base64_screenshot)
            with open(screenshot_path, 'wb') as f:
                f.write(image_data)

            # Verify the saved image (as done in utils.py)
            Image.open(screenshot_path).verify()

            # Load and check content is preserved
            loaded = Image.open(screenshot_path)
            assert loaded.size == (800, 600)
            assert loaded.getpixel((400, 25)) == (60, 60, 60)  # Header
            assert loaded.getpixel((150, 165)) == (0, 120, 215)  # Button

    def test_fallback_save_with_optimize(self):
        """Test fallback save method using PIL with optimize=True."""
        # Simulate receiving base64 screenshot data
        original = Image.new('RGB', (100, 100), color=(50, 100, 150))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_screenshot = base64.b64encode(buffer.getvalue()).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_path = Path(tmpdir) / 'screenshot.png'

            # Fallback method: Use png_base64_url_to_image + save with optimize
            image = png_base64_url_to_image(base64_screenshot)
            image.save(screenshot_path, format='PNG', optimize=True)

            # Verify content
            loaded = Image.open(screenshot_path)
            assert loaded.size == (100, 100)
            assert loaded.getpixel((50, 50)) == (50, 100, 150)

    def test_base64_with_data_prefix_workflow(self):
        """Test handling base64 with data URL prefix (browser format)."""
        original = Image.new('RGB', (640, 480), color=(100, 150, 200))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_with_prefix = (
            'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            screenshot_path = Path(tmpdir) / 'screenshot.png'

            # Extract base64 data (as done in utils.py)
            base64_data = base64_with_prefix.split(',')[1]
            image_data = base64.b64decode(base64_data)

            with open(screenshot_path, 'wb') as f:
                f.write(image_data)

            # Verify
            Image.open(screenshot_path).verify()
            loaded = Image.open(screenshot_path)
            assert loaded.size == (640, 480)


class TestEdgeCases:
    """Tests for edge cases in OpenHands integration functions."""

    def test_single_pixel_image(self):
        """Test conversion of 1x1 pixel image."""
        img = Image.new('RGB', (1, 1), color=(42, 84, 126))
        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (1, 1)
        assert result.getpixel((0, 0)) == (42, 84, 126)

    def test_grayscale_image(self):
        """Test conversion of grayscale image."""
        img = Image.new('L', (50, 50), color=128)

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (50, 50)

    def test_grayscale_numpy_array(self):
        """Test conversion of 2D grayscale numpy array."""
        img_array = np.full((50, 50), 128, dtype=np.uint8)

        base64_str = image_to_png_base64_url(img_array)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (50, 50)

    def test_palette_mode_image(self):
        """Test conversion of palette (P) mode image."""
        img_rgb = Image.new('RGB', (50, 50), color=(100, 150, 200))
        img_p = img_rgb.convert('P')

        base64_str = image_to_png_base64_url(img_p)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (50, 50)

    def test_high_resolution_image(self):
        """Test handling of high resolution images."""
        img = Image.new('RGB', (1920, 1080), color=(50, 100, 150))

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (1920, 1080)


class TestDataIntegrity:
    """Tests for data integrity through the integration functions."""

    def test_binary_data_preservation(self):
        """Ensure binary image data is preserved through encoding/decoding."""
        img = Image.new('RGB', (100, 100))
        pixels = img.load()
        for i in range(100):
            for j in range(100):
                pixels[i, j] = ((i + j) % 256, (i * 2) % 256, (j * 2) % 256)

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        result_pixels = result.load()
        for i in range(100):
            for j in range(100):
                expected = ((i + j) % 256, (i * 2) % 256, (j * 2) % 256)
                assert result_pixels[i, j] == expected

    def test_multiple_roundtrips(self):
        """Test that multiple roundtrip conversions preserve image quality."""
        img = Image.new('RGB', (100, 100), color=(128, 64, 196))

        current = img
        for _ in range(5):
            base64_str = image_to_png_base64_url(current)
            current = png_base64_url_to_image(base64_str)

        assert current.size == img.size
        assert current.getpixel((50, 50)) == (128, 64, 196)


class TestPillowVersion:
    """Tests to verify pillow version for CVE-2026-25990 fix."""

    def test_pillow_version_requirement(self):
        """Verify pillow version is 12.x or higher (CVE-2026-25990 fix)."""
        from PIL import __version__

        assert __version__ is not None
        major_version = int(__version__.split('.')[0])
        assert major_version >= 12, f'Expected pillow 12.x+, got {__version__}'
