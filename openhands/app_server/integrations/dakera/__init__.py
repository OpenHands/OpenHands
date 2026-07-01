"""Dakera memory integration for OpenHands.

This package wires Dakera's persistent, decay-weighted vector memory into
OpenHands conversations via the ConversationSecretEnricher extension point.

Configuration (all via environment variables):
    DAKERA_API_URL      – Dakera server base URL (default: http://localhost:3300)
    DAKERA_API_KEY      – Bearer token for the Dakera API (optional)
    DAKERA_AGENT_ID     – Agent identifier used for memory scoping (default: "openhands")
    DAKERA_TOP_K        – Number of memories to retrieve per turn (default: 5)
    DAKERA_ENABLED      – Set to "false" to disable (default: "true")

Usage (set in ServerConfig or as env var):
    OPENHANDS_CONFIG_CLS is not needed; set conversation_secret_enricher_class instead:

    server_config.conversation_secret_enricher_class = (
        "openhands.app_server.integrations.dakera.enricher.DakeraConversationSecretEnricher"
    )
"""
