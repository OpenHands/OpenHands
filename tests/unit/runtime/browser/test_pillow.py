"""Unit tests for Pillow (PIL) usage in OpenHands.

These tests ensure that pillow functionality works correctly across different
image formats, modes, and operations used in the browser module.
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
        # Create a simple RGB numpy array (100x100 red image)
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red channel

        result = image_to_png_base64_url(img_array)

        # Verify it's a valid base64 string
        assert isinstance(result, str)
        assert len(result) > 0

        # Verify we can decode it back to an image
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (100, 100)
        assert img.mode == 'RGB'

    def test_numpy_rgba_array_conversion(self):
        """Test conversion of RGBA numpy array to base64 PNG."""
        # Create a simple RGBA numpy array (50x50 with transparency)
        img_array = np.zeros((50, 50, 4), dtype=np.uint8)
        img_array[:, :, 0] = 255  # Red
        img_array[:, :, 3] = 128  # Half transparency

        result = image_to_png_base64_url(img_array)

        # Verify it's valid and can be decoded
        decoded = base64.b64decode(result)
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (50, 50)
        # Should be converted to RGB (mode conversion happens in the function)
        assert img.mode == 'RGB'

    def test_pil_image_conversion(self):
        """Test conversion of PIL Image to base64 PNG."""
        # Create a PIL Image
        img = Image.new('RGB', (200, 150), color=(0, 255, 0))  # Green image

        result = image_to_png_base64_url(img)

        # Verify we can decode it back
        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.size == (200, 150)
        assert decoded_img.mode == 'RGB'

    def test_rgba_pil_image_conversion(self):
        """Test conversion of RGBA PIL Image to base64 PNG."""
        # Create an RGBA PIL Image
        img = Image.new('RGBA', (75, 75), color=(0, 0, 255, 128))  # Blue with alpha

        result = image_to_png_base64_url(img)

        # Verify the result - RGBA is converted to RGB
        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.size == (75, 75)
        assert decoded_img.mode == 'RGB'

    def test_la_mode_conversion(self):
        """Test conversion of LA (grayscale with alpha) mode image."""
        # Create an LA mode image (grayscale + alpha)
        img = Image.new('LA', (80, 80), color=(128, 200))

        result = image_to_png_base64_url(img)

        # Verify conversion to RGB
        decoded = base64.b64decode(result)
        decoded_img = Image.open(io.BytesIO(decoded))
        assert decoded_img.mode == 'RGB'

    def test_data_prefix_option(self):
        """Test the add_data_prefix option."""
        img = Image.new('RGB', (10, 10), color=(100, 100, 100))

        # Without prefix (default)
        result_no_prefix = image_to_png_base64_url(img, add_data_prefix=False)
        assert not result_no_prefix.startswith('data:image/png;base64,')

        # With prefix
        result_with_prefix = image_to_png_base64_url(img, add_data_prefix=True)
        assert result_with_prefix.startswith('data:image/png;base64,')

        # The base64 content should be the same
        assert result_with_prefix.split(',')[1] == result_no_prefix

    def test_various_image_sizes(self):
        """Test conversion with various image sizes."""
        sizes = [(1, 1), (10, 10), (100, 100), (640, 480), (1920, 1080)]

        for width, height in sizes:
            img = Image.new('RGB', (width, height), color=(50, 100, 150))
            result = image_to_png_base64_url(img)

            # Verify round-trip
            decoded = base64.b64decode(result)
            decoded_img = Image.open(io.BytesIO(decoded))
            assert decoded_img.size == (width, height)


class TestPngBase64UrlToImage:
    """Tests for the png_base64_url_to_image function."""

    def test_base64_without_prefix(self):
        """Test decoding base64 string without data URL prefix."""
        # Create a test image and encode it
        original = Image.new('RGB', (50, 50), color=(255, 0, 0))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode()

        # Decode
        result = png_base64_url_to_image(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == (50, 50)

    def test_base64_with_data_prefix(self):
        """Test decoding base64 string with data URL prefix."""
        # Create a test image and encode it with prefix
        original = Image.new('RGB', (100, 75), color=(0, 255, 0))
        buffer = io.BytesIO()
        original.save(buffer, format='PNG')
        base64_str = (
            'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        )

        # Decode
        result = png_base64_url_to_image(base64_str)

        assert isinstance(result, Image.Image)
        assert result.size == (100, 75)

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion: image -> base64 -> image."""
        # Create original image with known content
        original = Image.new('RGB', (30, 30), color=(128, 64, 192))

        # Convert to base64 and back
        base64_str = image_to_png_base64_url(original)
        result = png_base64_url_to_image(base64_str)

        # Verify
        assert result.size == original.size
        assert result.mode == 'RGB'

        # Check pixel values match (sample a few pixels)
        orig_pixel = original.getpixel((15, 15))
        result_pixel = result.getpixel((15, 15))
        assert orig_pixel == result_pixel

    def test_roundtrip_with_drawing(self):
        """Test roundtrip with an image containing drawn content."""
        # Create image with drawn content
        original = Image.new('RGB', (200, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(original)
        draw.rectangle([10, 10, 50, 50], fill=(255, 0, 0))
        draw.ellipse([60, 10, 100, 50], fill=(0, 0, 255))

        # Roundtrip
        base64_str = image_to_png_base64_url(original)
        result = png_base64_url_to_image(base64_str)

        # Verify the drawn content is preserved
        # Check red rectangle area
        red_pixel = result.getpixel((30, 30))
        assert red_pixel == (255, 0, 0)

        # Check blue ellipse area
        blue_pixel = result.getpixel((80, 30))
        assert blue_pixel == (0, 0, 255)

        # Check white background
        white_pixel = result.getpixel((150, 50))
        assert white_pixel == (255, 255, 255)


class TestImageVerification:
    """Tests for image verification functionality used in screenshot saving."""

    def test_image_verify_valid_png(self):
        """Test that Image.verify() works on valid PNG files."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color=(100, 150, 200))
            img.save(f.name, format='PNG')

            # Verify the saved image
            verify_img = Image.open(f.name)
            verify_img.verify()  # Should not raise

            Path(f.name).unlink()

    def test_image_verify_with_optimize(self):
        """Test saving with optimize=True (used in fallback screenshot saving)."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color=(50, 100, 150))
            img.save(f.name, format='PNG', optimize=True)

            # Load and verify
            loaded = Image.open(f.name)
            assert loaded.size == (100, 100)

            # Verify content is preserved
            pixel = loaded.getpixel((50, 50))
            assert pixel == (50, 100, 150)

            Path(f.name).unlink()

    def test_image_save_various_formats(self):
        """Test that pillow can save images in various formats used by the system."""
        img = Image.new('RGB', (100, 100), color=(200, 100, 50))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test PNG (primary format used)
            png_path = Path(tmpdir) / 'test.png'
            img.save(png_path, format='PNG')
            assert png_path.exists()
            loaded_png = Image.open(png_path)
            assert loaded_png.format == 'PNG'

            # Test JPEG (for potential future use)
            jpg_path = Path(tmpdir) / 'test.jpg'
            img.save(jpg_path, format='JPEG')
            assert jpg_path.exists()
            loaded_jpg = Image.open(jpg_path)
            assert loaded_jpg.format == 'JPEG'


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_pixel_image(self):
        """Test conversion of 1x1 pixel image."""
        img = Image.new('RGB', (1, 1), color=(42, 84, 126))
        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (1, 1)
        assert result.getpixel((0, 0)) == (42, 84, 126)

    def test_grayscale_image(self):
        """Test conversion of grayscale image."""
        img = Image.new('L', (50, 50), color=128)  # Grayscale

        # Convert to base64 (will need conversion to RGB internally)
        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        # Should be able to decode successfully
        assert result.size == (50, 50)

    def test_grayscale_numpy_array(self):
        """Test conversion of 2D grayscale numpy array."""
        # 2D array (grayscale)
        img_array = np.full((50, 50), 128, dtype=np.uint8)

        base64_str = image_to_png_base64_url(img_array)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (50, 50)

    def test_palette_mode_image(self):
        """Test conversion of palette (P) mode image."""
        # Create RGB image first, then convert to palette
        img_rgb = Image.new('RGB', (50, 50), color=(100, 150, 200))
        img_p = img_rgb.convert('P')

        base64_str = image_to_png_base64_url(img_p)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (50, 50)

    def test_high_resolution_image(self):
        """Test handling of high resolution images."""
        # Create a 4K-like image (scaled down for test speed)
        img = Image.new('RGB', (1920, 1080), color=(50, 100, 150))

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (1920, 1080)

    def test_image_with_complex_content(self):
        """Test image with complex drawn content (simulating screenshot)."""
        # Simulate a browser screenshot with various UI elements
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Header bar
        draw.rectangle([0, 0, 800, 50], fill=(60, 60, 60))

        # Content area
        draw.rectangle([50, 100, 750, 550], fill=(240, 240, 240))

        # Button
        draw.rectangle([100, 150, 200, 180], fill=(0, 120, 215))

        # Text areas (represented as rectangles)
        draw.rectangle([100, 200, 700, 220], fill=(200, 200, 200))
        draw.rectangle([100, 240, 700, 260], fill=(200, 200, 200))

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        assert result.size == (800, 600)

        # Verify specific regions
        header_pixel = result.getpixel((400, 25))
        assert header_pixel == (60, 60, 60)

        button_pixel = result.getpixel((150, 165))
        assert button_pixel == (0, 120, 215)


class TestBase64EncodingDecoding:
    """Tests for base64 encoding/decoding edge cases."""

    def test_binary_data_preservation(self):
        """Ensure binary image data is preserved through base64 encoding."""
        # Create image with specific byte patterns
        img = Image.new('RGB', (100, 100))
        pixels = img.load()
        for i in range(100):
            for j in range(100):
                pixels[i, j] = ((i + j) % 256, (i * 2) % 256, (j * 2) % 256)

        base64_str = image_to_png_base64_url(img)
        result = png_base64_url_to_image(base64_str)

        # Verify pixel values match
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

        # After multiple roundtrips, image should still be the same
        # (PNG is lossless)
        assert current.size == img.size
        assert current.getpixel((50, 50)) == (128, 64, 196)


class TestPillowVersion:
    """Tests to verify pillow version compatibility."""

    def test_pillow_import(self):
        """Verify pillow can be imported successfully."""
        from PIL import __version__

        assert __version__ is not None
        # Verify we're using pillow 12.x or higher (CVE-2026-25990 fix)
        major_version = int(__version__.split('.')[0])
        assert major_version >= 12, f'Expected pillow 12.x+, got {__version__}'

    def test_pillow_features_available(self):
        """Test that required pillow features are available."""
        # Verify Image.fromarray works (used in base64.py)
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        assert img.size == (10, 10)

        # Verify Image.open works with BytesIO (used in base64.py)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        loaded = Image.open(buffer)
        assert loaded.size == (10, 10)

        # Verify Image.verify works (used in utils.py)
        buffer.seek(0)
        verify_img = Image.open(buffer)
        verify_img.verify()  # Should not raise

        # Verify mode conversion works (used in base64.py)
        rgba = Image.new('RGBA', (10, 10))
        rgb = rgba.convert('RGB')
        assert rgb.mode == 'RGB'
