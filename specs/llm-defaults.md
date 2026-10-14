# LLM Defaults Specs

---

### LLD-001: Frontend always sends its chosen default model

- [x] When the agent-server returns an absent or empty `llm.model` (e.g. because the user has never saved settings), the frontend adapter shall substitute `DEFAULT_SETTINGS.llm_model` (`"openai/gpt-5.6-sol"`) before sending a conversation-start request that carries a raw agent spec (the local planner). Agent Profile launches take the model from the referenced LLM profile.
- [x] The frontend shall never rely on the agent-server SDK's own default model (`gpt-5.5`); it shall always send an explicit model value.
- [x] Whitespace-only model strings shall be treated as absent and fall back to the default.
- [x] The same guard applies when LLM settings arrive via `encryptedAgentSettings` (the planner's encrypted payload path).
