import React from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";

import {
  buildNewSessionName,
  isValidAgentName,
  type NewMarsAgentInput,
} from "#/api/mars/mars-tunnel-backend";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const DEFAULT_NAME_STEM = "openhands";
const NAME_HINT_ID = "digitalocean-new-agent-name-hint";
const LLM_KEY_HINT_ID = "digitalocean-new-agent-llm-key-hint";
const HELPER_CLASS = "-mt-1 text-pretty text-xs text-[var(--oh-muted)]";

interface DigitalOceanNewAgentFormProps {
  isCreating: boolean;
  error: string | null;
  onSubmit: (input: NewMarsAgentInput) => void;
  onCancel: () => void;
}

/** Name and optional LLM key for a new OpenHands agent on DigitalOcean. */
export function DigitalOceanNewAgentForm({
  isCreating,
  error,
  onSubmit,
  onCancel,
}: DigitalOceanNewAgentFormProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = React.useState(() =>
    buildNewSessionName(DEFAULT_NAME_STEM),
  );
  const [llmApiKey, setLlmApiKey] = React.useState("");
  const trimmedName = name.trim();
  const isNameValid = isValidAgentName(trimmedName);
  const showNameError = trimmedName !== "" && !isNameValid;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!isNameValid || isCreating) return;
    onSubmit({
      name: trimmedName,
      llmApiKey: llmApiKey.trim() || undefined,
    });
  };

  return (
    <form
      onSubmit={submit}
      data-testid="digitalocean-new-agent-form"
      className="flex flex-col gap-4 border-b border-[var(--oh-border)] px-4 pb-4 pt-1"
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsInput
            testId="digitalocean-new-agent-name"
            name="digitalocean-new-agent-name"
            type="text"
            label={t(I18nKey.DO_AGENTS$NEW_AGENT_NAME_LABEL)}
            value={name}
            onChange={setName}
            isDisabled={isCreating}
            ariaInvalid={showNameError}
            ariaDescribedBy={NAME_HINT_ID}
            inputClassName={cn(showNameError && "border-red-500")}
          />
          <p
            id={NAME_HINT_ID}
            className={cn(HELPER_CLASS, showNameError && "text-red-400")}
          >
            {t(I18nKey.DO_AGENTS$NEW_AGENT_NAME_HINT)}
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <SettingsInput
            testId="digitalocean-new-agent-llm-key"
            name="digitalocean-new-agent-llm-key"
            type="password"
            label={t(I18nKey.DO_AGENTS$NEW_AGENT_LLM_KEY_LABEL)}
            value={llmApiKey}
            onChange={setLlmApiKey}
            isDisabled={isCreating}
            showOptionalTag
            ariaDescribedBy={LLM_KEY_HINT_ID}
          />
          <p id={LLM_KEY_HINT_ID} className={HELPER_CLASS}>
            {t(I18nKey.DO_AGENTS$NEW_AGENT_LLM_KEY_HINT)}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <p
          role={error ? "alert" : undefined}
          data-testid={error ? "digitalocean-new-agent-error" : undefined}
          className="min-w-0 flex-1 text-pretty text-xs text-[var(--oh-status-error)]"
        >
          {error}
        </p>
        <BrandButton
          type="button"
          variant="secondary"
          onClick={onCancel}
          isDisabled={isCreating}
          testId="digitalocean-new-agent-cancel"
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton
          type="submit"
          variant="primary"
          isDisabled={!isNameValid || isCreating}
          testId="digitalocean-new-agent-submit"
          startContent={
            isCreating ? (
              <Loader2 className="size-4 animate-spin" aria-hidden />
            ) : null
          }
        >
          {isCreating
            ? t(I18nKey.DO_AGENTS$CREATING_AGENT)
            : t(I18nKey.DO_AGENTS$NEW_AGENT_CREATE)}
        </BrandButton>
      </div>
    </form>
  );
}
