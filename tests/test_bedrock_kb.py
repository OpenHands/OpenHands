"""Tests for Bedrock Knowledge Base integrations."""

from unittest.mock import MagicMock


class TestBedrockKnowledgeBase:
    def test_get_context_returns_formatted_passages(self):
        from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            'retrievalResults': [
                {
                    'content': {'text': 'Doc 1'},
                    'location': {'s3Location': {'uri': 's3://b/1'}},
                    'score': 0.9,
                },
                {
                    'content': {'text': 'Doc 2'},
                    'location': {'s3Location': {'uri': 's3://b/2'}},
                    'score': 0.8,
                },
            ]
        }
        kb = BedrockKnowledgeBase(
            knowledge_base_id='TEST123', use_agentic_retrieval=False
        )
        kb._client = mock_client
        context = kb.get_context('test query')
        assert 'Doc 1' in context
        assert 'Doc 2' in context
        assert 's3://b/1' in context

    def test_get_context_empty_kb_id(self):
        from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

        kb = BedrockKnowledgeBase(knowledge_base_id='')
        assert kb.get_context('test') == ''

    def test_search_returns_structured_results(self):
        from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            'retrievalResults': [
                {
                    'content': {'text': 'Result'},
                    'location': {'s3Location': {'uri': 's3://b/r'}},
                    'score': 0.95,
                },
            ]
        }
        kb = BedrockKnowledgeBase(
            knowledge_base_id='TEST123', use_agentic_retrieval=False
        )
        kb._client = mock_client
        results = kb.search('query')
        assert len(results) == 1
        assert results[0]['content'] == 'Result'
        assert results[0]['score'] == 0.95

    def test_uses_managed_search_config(self):
        from openhands.knowledge.bedrock_knowledge_base import BedrockKnowledgeBase

        mock_client = MagicMock()
        mock_client.retrieve.return_value = {'retrievalResults': []}
        kb = BedrockKnowledgeBase(
            knowledge_base_id='TEST123', use_agentic_retrieval=False
        )
        kb._client = mock_client
        kb.get_context('test')
        call_kwargs = mock_client.retrieve.call_args.kwargs
        assert 'managedSearchConfiguration' in call_kwargs['retrievalConfiguration']
