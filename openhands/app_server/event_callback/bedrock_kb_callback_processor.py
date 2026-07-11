"""Amazon Bedrock Knowledge Base callback processor for OpenHands.

Enriches conversation context by retrieving relevant documentation from
a Bedrock Managed Knowledge Base when user messages are received.

Configuration:
    Set these environment variables:
        KNOWLEDGE_BASE_ID: The Bedrock Knowledge Base ID
        AWS_REGION: AWS region (default: us-east-1)
        BEDROCK_KB_MAX_RESULTS: Max results to retrieve (default: 5)

Registration:
    Add to your event callback configuration to automatically enrich
    conversations with KB context on each user message.
"""

import logging
import os
from typing import ClassVar, Optional
from uuid import UUID

from openhands.app_server.event_callback.event_callback_models import (
    EventCallback,
    EventCallbackProcessor,
    EventCallbackResult,
    EventCallbackResultStatus,
    EventKind,
)

logger = logging.getLogger(__name__)


class BedrockKBCallbackProcessor(EventCallbackProcessor):
    """Enriches conversation context with Bedrock Knowledge Base retrieval.

    On each user message event, queries the configured KB and appends
    relevant documentation as context for the agent.
    """

    event_kind: ClassVar[EventKind] = 'MessageEvent'

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        region_name: Optional[str] = None,
        number_of_results: int = 5,
    ):
        self.knowledge_base_id = knowledge_base_id or os.environ.get(
            'KNOWLEDGE_BASE_ID', ''
        )
        self.region_name = region_name or os.environ.get('AWS_REGION', 'us-east-1')
        self.number_of_results = int(
            os.environ.get('BEDROCK_KB_MAX_RESULTS', number_of_results)
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ImportError(
                    'boto3 is required for BedrockKBCallbackProcessor. '
                    'Install with: pip install boto3>=1.43.2'
                )
            self._client = boto3.client(
                'bedrock-agent-runtime',
                region_name=self.region_name,
                config=Config(user_agent_extra='openhands/bedrock-kb'),
            )
        return self._client

    async def __call__(
        self,
        conversation_id: UUID,
        callback: EventCallback,
        event,
    ) -> EventCallbackResult | None:
        """Process a message event by enriching with KB context."""
        if not self.knowledge_base_id:
            return None

        # Extract user message content
        query = self._extract_query(event)
        if not query:
            return None

        # Retrieve from KB
        context = self._retrieve(query)
        if not context:
            return None

        # Return enriched context as a result
        logger.info(
            f'Enriched conversation {conversation_id} with {len(context)} KB passages'
        )

        return EventCallbackResult(
            status=EventCallbackResultStatus.SUCCESS,
            event_callback_id=callback.id,
            event_id=event.id,
            conversation_id=conversation_id,
            detail=context,
        )

    def _extract_query(self, event) -> str:
        """Extract the query text from a message event."""
        if hasattr(event, 'content'):
            content = event.content
            if isinstance(content, str):
                return content
            elif isinstance(content, dict):
                return content.get('text', '')
        if hasattr(event, 'message'):
            return str(event.message)
        return ''

    def _retrieve(self, query: str) -> str:
        """Retrieve relevant context from the KB."""
        retrieval_config = {
            'managedSearchConfiguration': {'numberOfResults': self.number_of_results}
        }

        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration=retrieval_config,
            )

            results = response.get('retrievalResults', [])
            if not results:
                return ''

            passages = []
            for result in results:
                content = result.get('content', {}).get('text', '')
                source = result.get('location', {}).get('s3Location', {}).get('uri', '')
                if content:
                    passages.append(f'{content}\n[Source: {source}]')

            return '\n\n---\n\n'.join(passages)
        except Exception as e:
            logger.error(f'Error retrieving from Bedrock KB: {e}')
            return ''
