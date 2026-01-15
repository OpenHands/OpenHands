# DEPRECATED: This module is part of the deprecated 'openhands.runtime' package.
# It will be removed on April 1, 2025. Please migrate to the OpenHands SDK:
# https://github.com/All-Hands-AI/openhands-sdk
import base64
import io

import numpy as np
from PIL import Image


def image_to_png_base64_url(
    image: np.ndarray | Image.Image, add_data_prefix: bool = False
) -> str:
    """Convert a numpy array to a base64 encoded png image url."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ('RGBA', 'LA'):
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')

    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return (
        f'data:image/png;base64,{image_base64}'
        if add_data_prefix
        else f'{image_base64}'
    )


def png_base64_url_to_image(png_base64_url: str) -> Image.Image:
    """Convert a base64 encoded png image url to a PIL Image."""
    splited = png_base64_url.split(',')
    if len(splited) == 2:
        base64_data = splited[1]
    else:
        base64_data = png_base64_url
    return Image.open(io.BytesIO(base64.b64decode(base64_data)))
