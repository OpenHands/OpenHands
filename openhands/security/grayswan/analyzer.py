"""GraySwan security analyzer for OpenHands."""

import asyncio
import os
from typing import Any

import aiohttp
from fastapi import Request

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action, ActionSecurityRisk
from openhands.events.event_store_abc import EventStoreABC
from openhands.memory.view import View
from openhands.security.analyzer import SecurityAnalyzer
from openhands.security.grayswan.utils import convert_events_to_openai_messages


class GraySwanAnalyzer(SecurityAnalyzer):
    """Security analyzer using GraySwan's Cygnal API for AI safety monitoring."""

    def __init__(
        self,
        history_limit: int = 20,
        max_message_chars: int = 30000,
        timeout: int = 30,
        low_threshold: float = 0.3,
        medium_threshold: float = 0.7,
        high_threshold: float = 1.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Initialize GraySwan analyzer.

        Args:
            history_limit: Number of recent events to include as context
            max_message_chars: Max characters for conversation processing
            timeout: Request timeout in seconds
            low_threshold: Risk threshold for LOW classification (default: 0.3)
            medium_threshold: Risk threshold for MEDIUM classification (default: 0.7)
            high_threshold: Risk threshold for HIGH classification (default: 1.0)
            session: Optional pre-configured session (mainly for testing)

        Environment Variables:
            GRAYSWAN_API_KEY: Required API key for GraySwan authentication
            GRAYSWAN_POLICY_ID: Optional policy ID for custom GraySwan policy
        """
        super().__init__()

        self.api_key = os.getenv('GRAYSWAN_API_KEY')
        if not self.api_key:
            logger.error(
                'GRAYSWAN_API_KEY environment variable is required for GraySwanAnalyzer'
            )
            raise ValueError('GRAYSWAN_API_KEY environment variable is required')

        self.policy_id = os.getenv('GRAYSWAN_POLICY_ID')
        if not self.policy_id:
            self.policy_id = (
                '689ca4885af3538a39b2ba04'  # GraySwan default coding agent policy
            )
            logger.info(f'Using default GraySwan policy ID: {self.policy_id}')
        else:
            logger.info(f'Using GraySwan policy ID from environment: {self.policy_id}')

        self.event_stream: EventStoreABC | None = None
        self.history_limit = history_limit
        self.max_message_chars = max_message_chars
        self.timeout = timeout

        self.violation_thresholds = {
            'low': low_threshold,
            'medium': medium_threshold,
            'high': high_threshold,
        }

        self.api_url = 'https://api.grayswan.ai/cygnal/monitor'
        self.session: aiohttp.ClientSession | None = session

        logger.info(
            f'GraySwanAnalyzer initialized with history_limit={history_limit}, timeout={timeout}s'
        )

    def set_event_stream(self, event_stream: EventStoreABC) -> None:
        """Set the event stream for accessing conversation history."""
        self.event_stream = event_stream
        logger.debug('Event stream set for GraySwanAnalyzer')

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'x-grayswan-api-key': self.api_key,
                    'Content-Type': 'application/json',
                },
            )
        elif hasattr(self.session, 'closed') and self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'x-grayswan-api-key': self.api_key,
                    'Content-Type': 'application/json',
                },
            )
        return self.session

    def _map_violation_to_risk(self, violation_score: float) -> ActionSecurityRisk:
        """Map GraySwan violation score to OpenHands ActionSecurityRisk."""
        if violation_score <= self.violation_thresholds['low']:
            return ActionSecurityRisk.LOW
        elif violation_score <= self.violation_thresholds['medium']:
            return ActionSecurityRisk.MEDIUM
        else:
            return ActionSecurityRisk.HIGH

    async def _call_grayswan_api(
        self, messages: list[dict[str, Any]]
    ) -> ActionSecurityRisk:
        """Call GraySwan API with formatted messages."""
        try:
            session = await self._get_session()

            payload = {'messages': messages, 'policy_id': self.policy_id}

            logger.info(
                f'Sending request to GraySwan API with {len(messages)} messages and policy_id: {self.policy_id}'
            )
            logger.info(f'Payload: {payload}')

            response = await session.post(self.api_url, json=payload)

            async with response as resp:
                if resp.status == 200:
                    result = await resp.json()
                    violation_score = result.get('violation', 0.0)
                    risk_level = self._map_violation_to_risk(violation_score)
                    if 'ipi' in result and result['ipi']:
                        risk_level = (
                            ActionSecurityRisk.HIGH
                        )  # indirect prompt injection is auto escalated to HIGH
                    logger.info(
                        f'GraySwan risk assessment: {risk_level.name} (violation_score: {violation_score:.2f})'
                    )
                    return risk_level
                else:
                    error_text = await resp.text()
                    logger.error(f'GraySwan API error {resp.status}: {error_text}')
                    return ActionSecurityRisk.UNKNOWN

        except asyncio.TimeoutError:
            logger.error('GraySwan API request timed out')
            return ActionSecurityRisk.UNKNOWN
        except Exception as e:
            logger.error(f'GraySwan security analysis failed: {e}')
            return ActionSecurityRisk.UNKNOWN

    async def handle_api_request(self, request: Request) -> Any:
        """Handle incoming API requests for configuration or webhooks."""
        return {'status': 'ok', 'analyzer': 'grayswan'}

    async def security_risk(self, action: Action) -> ActionSecurityRisk:
        """Analyze action for security risks using GraySwan API."""
        logger.debug(
            f'Calling security_risk on GraySwanAnalyzer for action: {type(action).__name__}'
        )

        if not self.event_stream:
            logger.warning('No event stream available for GraySwan analysis')
            return ActionSecurityRisk.UNKNOWN

        try:
            # Use View to get closer to what the agent's LLM actually sees
            # This applies context management (trimming, summaries, masking)
            view = View.from_events(list(self.event_stream.get_events()))
            recent_events = (
                list(view)[-self.history_limit :]
                if len(view) > self.history_limit
                else list(view)
            )

            events_to_process = recent_events + [action]
            openai_messages = convert_events_to_openai_messages(events_to_process)

            if not openai_messages:
                logger.warning('No valid messages to analyze')
                return ActionSecurityRisk.UNKNOWN

            logger.debug(
                f'Converted {len(events_to_process)} events into {len(openai_messages)} OpenAI messages for GraySwan analysis'
            )
            return await self._call_grayswan_api(openai_messages)

        except Exception as e:
            logger.error(f'GraySwan security analysis failed: {e}')
            return ActionSecurityRisk.UNKNOWN

    async def close(self) -> None:
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
