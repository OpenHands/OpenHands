export interface SettingsFieldConstraints {
  min?: number;
  max?: number;
  step?: number;
}

interface SettingsFieldMetadata {
  label?: string;
  description?: string;
  constraints?: SettingsFieldConstraints;
}

const looksLikeTranslationKey = (value: string | null | undefined) =>
  Boolean(value?.includes("$"));

const FIELD_METADATA: Record<string, SettingsFieldMetadata> = {
  agent: {
    label: "SETTINGS$AGENT",
    description: "SETTINGS$AGENT_TOOLTIP",
  },
  max_iterations: {
    description:
      "Maximum number of agent steps allowed before the conversation stops.",
  },
  confirmation_mode: {
    label: "SETTINGS_FORM$ENABLE_CONFIRMATION_MODE_LABEL",
    description: "SETTINGS$CONFIRMATION_MODE_TOOLTIP",
  },
  security_analyzer: {
    label: "SETTINGS_FORM$SECURITY_ANALYZER_LABEL",
    description: "SETTINGS$SECURITY_ANALYZER_DESCRIPTION",
  },
  "critic.enabled": {
    description: "Enable the critic service to review the agent's progress.",
  },
  "critic.mode": {
    description: "Choose when the critic should review and intervene.",
  },
  "llm.top_p": {
    label: "Top P",
    description:
      "Controls nucleus sampling by limiting token selection to the smallest set whose cumulative probability exceeds this value.",
    constraints: {
      min: 0,
      max: 1,
      step: 0.01,
    },
  },
  "llm.temperature": {
    label: "Temperature",
    description:
      "Controls randomness in model responses. Lower values are more deterministic.",
    constraints: {
      min: 0,
      max: 2,
      step: 0.1,
    },
  },
  "llm.reasoning_effort": {
    label: "Reasoning effort",
    description:
      "Controls how much effort supported reasoning models spend thinking before responding.",
  },
  "llm.max_input_tokens": {
    label: "Max input tokens",
    description:
      "Sets the maximum number of prompt tokens allowed for the selected model when supported.",
  },
  "llm.max_output_tokens": {
    label: "Max output tokens",
    description:
      "Sets the maximum number of tokens the model may generate in a response.",
  },
};

export function getSettingsFieldMetadata(fieldKey: string) {
  return FIELD_METADATA[fieldKey];
}

export function getSettingsFieldLabel(fieldKey: string, fallback: string) {
  if (looksLikeTranslationKey(fallback)) {
    return fallback;
  }

  return FIELD_METADATA[fieldKey]?.label ?? fallback;
}

export function getSettingsFieldDescription(
  fieldKey: string,
  fallback?: string | null,
) {
  if (looksLikeTranslationKey(fallback)) {
    return fallback;
  }

  return FIELD_METADATA[fieldKey]?.description ?? fallback ?? null;
}

export function getSettingsFieldConstraints(fieldKey: string) {
  return FIELD_METADATA[fieldKey]?.constraints;
}
