"""
Cookie compression utilities for keycloak_auth cookie.

This module provides functions to compress and decompress cookie data
to reduce cookie size and improve performance.
"""

import base64
import gzip

from openhands.core.logger import openhands_logger as logger


def compress_cookie_data(data: str) -> str:
    """
    Compress cookie data using gzip and encode with base64.

    Args:
        data: The cookie data string to compress

    Returns:
        Base64 encoded compressed data with 'gz:' prefix to indicate compression

    Raises:
        Exception: If compression fails
    """
    try:
        # Convert string to bytes
        data_bytes = data.encode('utf-8')

        # Compress using gzip
        compressed_bytes = gzip.compress(data_bytes, compresslevel=6)

        # Encode with base64 for safe cookie storage
        encoded_data = base64.b64encode(compressed_bytes).decode('ascii')

        # Add prefix to indicate this is compressed data
        compressed_cookie = f'gz:{encoded_data}'

        logger.debug(
            'Cookie compression stats',
            extra={
                'original_size': len(data),
                'compressed_size': len(compressed_cookie),
                'compression_ratio': len(compressed_cookie) / len(data)
                if len(data) > 0
                else 0,
            },
        )

        return compressed_cookie

    except Exception as e:
        logger.error(f'Failed to compress cookie data: {str(e)}')
        raise


def decompress_cookie_data(data: str) -> str:
    """
    Decompress cookie data if it's compressed, otherwise return as-is.

    Args:
        data: The cookie data string (may be compressed or uncompressed)

    Returns:
        Decompressed cookie data string

    Raises:
        Exception: If decompression fails for compressed data
    """
    try:
        # Check if data is compressed (has 'gz:' prefix)
        if not data.startswith('gz:'):
            # Not compressed, return as-is for backward compatibility
            logger.debug('Cookie data is not compressed, returning as-is')
            return data

        # Remove the 'gz:' prefix
        encoded_data = data[3:]

        # Check for empty compressed data
        if not encoded_data:
            raise ValueError('Empty compressed data')

        # Decode from base64
        compressed_bytes = base64.b64decode(encoded_data.encode('ascii'))

        # Decompress using gzip
        decompressed_bytes = gzip.decompress(compressed_bytes)

        # Convert back to string
        decompressed_data = decompressed_bytes.decode('utf-8')

        logger.debug(
            'Cookie decompression stats',
            extra={
                'compressed_size': len(data),
                'decompressed_size': len(decompressed_data),
                'compression_ratio': len(data) / len(decompressed_data)
                if len(decompressed_data) > 0
                else 0,
            },
        )

        return decompressed_data

    except Exception as e:
        logger.error(f'Failed to decompress cookie data: {str(e)}')
        raise


def should_compress_cookie(data: str, min_size_threshold: int = 1000) -> bool:
    """
    Determine if cookie data should be compressed based on size.

    Args:
        data: The cookie data string
        min_size_threshold: Minimum size in bytes to consider compression

    Returns:
        True if data should be compressed, False otherwise
    """
    return len(data.encode('utf-8')) >= min_size_threshold
