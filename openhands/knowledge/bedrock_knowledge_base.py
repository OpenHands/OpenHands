"""Amazon Bedrock Knowledge Base context provider for OpenHands.

Provides relevant documentation context from a Bedrock Managed Knowledge Base
to inform code generation and task resolution.

Integration point: Use as a context source in OpenHands' event callback system
or as a pre-prompt context injector in custom agent configurations.

Usage (standalone):
    from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

    kb = BedrockKnowledgeBase(knowledge_base_id="ABCDEFGHIJ")
    context = kb.get_context("How do I implement retry logic?")

Usage (as event callback context enrichment):
    from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

    kb = BedrockKnowledgeBase(
        knowledge_base_id="ABCDEFGHIJ",
        region_name="us-west-2",
    )

    # In your event callback processor, enrich the prompt before sending to LLM:
    async def enrich_with_kb(event):
        if event.type == "user_message":
            kb_context = kb.get_context(event.content)
            if kb_context:
                event.system_context += f"\\n\\nRelevant documentation:\\n{kb_context}"
        return event

Environment variables:
    KNOWLEDGE_BASE_ID: The Bedrock KB ID
    AWS_REGION: AWS region (default: us-east-1)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get('location', {})
    loc_type = location.get('type', '')
    if loc_type == 'S3' or 's3Location' in location:
        return location.get('s3Location', {}).get('uri', '')
    elif loc_type == 'WEB' or 'webLocation' in location:
        return location.get('webLocation', {}).get('url', '')
    elif 'confluenceLocation' in location:
        return location.get('confluenceLocation', {}).get('url', '')
    elif 'salesforceLocation' in location:
        return location.get('salesforceLocation', {}).get('url', '')
    elif 'sharePointLocation' in location:
        return location.get('sharePointLocation', {}).get('url', '')
    elif 'customDocumentLocation' in location:
        return location.get('customDocumentLocation', {}).get('id', '')
    # Fallback to metadata._source_uri (for agentic results)
    return result.get('metadata', {}).get('_source_uri', '')


class BedrockKnowledgeBase:
    """Retrieves context from an Amazon Bedrock Managed Knowledge Base.

    Useful for providing internal documentation, coding standards, or
    project-specific knowledge to the coding agent.

    Args:
        knowledge_base_id: The KB ID. Falls back to KNOWLEDGE_BASE_ID env var.
        region_name: AWS region. Falls back to AWS_REGION env var or us-east-1.
        number_of_results: Max results to return. Defaults to 5.
        use_agentic_retrieval: If True, try AgenticRetrieveStream first with fallback to plain Retrieve.
    """

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        region_name: Optional[str] = None,
        number_of_results: int = 5,
        use_agentic_retrieval: Optional[bool] = None,
    ):
        self.knowledge_base_id = knowledge_base_id or os.environ.get(
            'KNOWLEDGE_BASE_ID', ''
        )
        self.region_name = region_name or os.environ.get('AWS_REGION', 'us-east-1')
        self.number_of_results = number_of_results
        self.use_agentic_retrieval = (
            use_agentic_retrieval
            if use_agentic_retrieval is not None
            else os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false'
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
                    'boto3 is required for Bedrock Knowledge Base. '
                    'Install with: pip install boto3>=1.43.2'
                )
            self._client = boto3.client(
                'bedrock-agent-runtime',
                region_name=self.region_name,
                config=Config(user_agent_extra='openhands/bedrock-kb'),
            )
        return self._client

    def _agentic_retrieve(self, query: str, top_k: int):
        """Try agentic retrieval with streaming. Returns list of passages or None on failure."""
        try:
            response = self.client.agentic_retrieve_stream(
                knowledgeBaseId=self.knowledge_base_id,
                messages=[{'content': {'text': query}, 'role': 'user'}],
                retrievers=[
                    {
                        'configuration': {
                            'knowledgeBase': {
                                'knowledgeBaseId': self.knowledge_base_id,
                                'retrievalOverrides': {'maxNumberOfResults': top_k},
                            }
                        }
                    }
                ],
                agenticRetrieveConfiguration={
                    'foundationModelType': 'MANAGED',
                    'rerankingModelType': 'MANAGED',
                },
            )
            passages = []
            for event in response.get('stream', []):
                if 'result' in event and 'results' in event['result']:
                    for result in event['result']['results']:
                        passages.append(
                            {
                                'content': result.get('content', {}).get('text', ''),
                                'source': _get_source_uri(result),
                                'score': result.get('score', 0.0),
                            }
                        )
            return passages
        except Exception as e:
            logger.debug(
                f'Agentic retrieval unavailable, will fall back to managed retrieve: {e}'
            )
            return None

    def get_context(self, query: str, max_results: Optional[int] = None) -> str:
        """Retrieve relevant context for a query.

        Args:
            query: The question or topic to find context for.
            max_results: Override the default number of results.

        Returns:
            Formatted string with relevant documentation passages.
        """
        if not self.knowledge_base_id:
            logger.warning('No knowledge_base_id configured. Skipping KB context.')
            return ''

        k = max_results or self.number_of_results

        # Try agentic retrieval first
        if self.use_agentic_retrieval:
            agentic_results = self._agentic_retrieve(query, k)
            if agentic_results is not None:
                if not agentic_results:
                    return ''
                passages = []
                for r in agentic_results:
                    if r['content']:
                        passages.append(f'{r["content"]}\n[Source: {r["source"]}]')
                return '\n\n---\n\n'.join(passages)

        # Fallback to managed/vector retrieve

        retrieval_config = {'managedSearchConfiguration': {'numberOfResults': k}}

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
                source = _get_source_uri(result)
                if content:
                    passages.append(f'{content}\n[Source: {source}]')

            return '\n\n---\n\n'.join(passages)
        except Exception as e:
            logger.error(f'Error retrieving from Bedrock KB: {e}')
            return ''

    def search(self, query: str, max_results: Optional[int] = None) -> list[dict]:
        """Search and return structured results.

        Args:
            query: The search query.
            max_results: Override default number of results.

        Returns:
            List of dicts with content, source, and score.
        """
        if not self.knowledge_base_id:
            return []

        k = max_results or self.number_of_results

        # Try agentic retrieval first
        if self.use_agentic_retrieval:
            agentic_results = self._agentic_retrieve(query, k)
            if agentic_results is not None:
                return agentic_results

        # Fallback to managed/vector retrieve

        retrieval_config = {'managedSearchConfiguration': {'numberOfResults': k}}

        try:
            response = self.client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration=retrieval_config,
            )

            results = []
            for result in response.get('retrievalResults', []):
                results.append(
                    {
                        'content': result.get('content', {}).get('text', ''),
                        'source': _get_source_uri(result),
                        'score': result.get('score', 0.0),
                    }
                )
            return results
        except Exception as e:
            logger.error(f'Error searching Bedrock KB: {e}')
            return []
