# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
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
