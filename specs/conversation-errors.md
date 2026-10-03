# Conversation error specs

### CE-001: Conversation errors remain readable in chat history

- [x] Chat renders `ConversationErrorEvent` from live events and loaded history using the existing expandable error presentation.
- [x] Expanding a conversation error displays its `detail` without requiring an `llm_message` field.
- [x] `AgentErrorEvent` continues to display its `error` field.

### CE-002: Transcript exports retain conversation error details

- [x] Markdown and HTML transcripts include `ConversationErrorEvent.detail` as an error entry using the existing export escaping rules.
