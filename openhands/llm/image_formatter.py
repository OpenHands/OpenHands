"""LLM-specific image formatting for multimodal messages.

Different LLMs expect images in different formats within messages:
- OpenAI/GPT-4V: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
- Claude: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
- Gemini: Uses google.genai.types with inline_data blobs

This module provides formatters to convert MCPImage objects to the appropriate format.
"""

from typing import Any

from openhands.events.observation.mcp import MCPImage


class LLMImageFormatter:
    """Format images for different LLM providers."""

    @staticmethod
    def format_for_openai(images: list[MCPImage]) -> list[dict[str, Any]]:
        """Format images for OpenAI/GPT-4V style APIs.

        Returns content blocks that can be added to a message's content array.
        """
        return [
            {
                'type': 'image_url',
                'image_url': {
                    'url': img.to_data_uri(),
                    'detail': 'auto',
                },
            }
            for img in images
        ]

    @staticmethod
    def format_for_claude(images: list[MCPImage]) -> list[dict[str, Any]]:
        """Format images for Anthropic Claude style APIs.

        Returns content blocks that can be added to a message's content array.
        """
        return [
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': img.mime_type,
                    'data': img.data,
                },
            }
            for img in images
        ]

    @staticmethod
    def format_for_gemini(images: list[MCPImage]) -> list[dict[str, Any]]:
        """Format images for Google Gemini style APIs.

        Returns dicts that can be converted to google.genai.types.Part.
        Note: Actual Gemini integration may need to import google.genai.types.
        """
        return [
            {
                'inline_data': {
                    'mime_type': img.mime_type,
                    'data': img.data,
                },
            }
            for img in images
        ]

    @staticmethod
    def format_for_litellm(images: list[MCPImage], model: str) -> list[dict[str, Any]]:
        """Auto-detect format based on model name (for LiteLLM compatibility).

        LiteLLM typically normalizes to OpenAI format, but some providers
        may need special handling.
        """
        model_lower = model.lower()

        if 'claude' in model_lower or 'anthropic' in model_lower:
            return LLMImageFormatter.format_for_claude(images)
        elif 'gemini' in model_lower or 'google' in model_lower:
            return LLMImageFormatter.format_for_gemini(images)
        else:
            # Default to OpenAI format (works for most via LiteLLM)
            return LLMImageFormatter.format_for_openai(images)

