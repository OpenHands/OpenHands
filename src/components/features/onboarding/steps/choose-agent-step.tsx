import React from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { Check } from "lucide-react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import {
  ACP_PROVIDER_FALLBACK_ICON,
  ACP_PROVIDERS,
  buildAcpAgentSettingsDiff,
} from "#/constants/acp-providers";
import {
  AgentBrandIcon,
  type AgentBrandIconKind,
} from "#/components/shared/agent-brand-icon";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

export type OnboardingAgentId =
  | "openhands"
  | "claude-code"
  | "codex"
  | "gemini-cli";

function getAgentOptionIcon(id: string): AgentBrandIconKind {
  if (id === "openhands") return "openhands";

  return (
    ACP_PROVIDERS.find(({ key }) => key === id)?.icon ??
    ACP_PROVIDER_FALLBACK_ICON
  );
}

export function AgentOptionIcon({ id, muted }: { id: string; muted: boolean }) {
  const icon = getAgentOptionIcon(id);

  // The OpenHands wordmark is wider than the square brand marks (24×16 vs
  // 18×18) and dims via opacity rather than a muted text colour — its paths
  // inherit ``currentColor`` so it stays white on the tile.
  if (icon === "openhands") {
    return (
      <AgentBrandIcon
        kind="openhands"
        size={16}
        className={cn("text-white", muted && "opacity-55")}
        data-testid="onboarding-agent-icon-openhands"
      />
    );
  }

  return (
    <AgentBrandIcon
      kind={icon}
      size={18}
      className={muted ? "text-[var(--oh-muted)]" : "text-white"}
      data-testid={`onboarding-agent-icon-${icon}`}
    />
  );
}

interface AgentOption {
  id: OnboardingAgentId;
  label: string;
  descriptionKey: I18nKey;
}

// Onboarding tile list is *derived* from the ACP registry so adding a
// new provider (or changing a display name) only needs one edit in
// ``acp-providers.ts``. The OpenHands tile is the only synthetic
// entry — it isn't an ACP provider, just the canonical default.
function getAgentOptions(): AgentOption[] {
  return [
    {
      id: "openhands",
      label: "OpenHands",
      descriptionKey: I18nKey.ONBOARDING$AGENT_OPENHANDS_DESCRIPTION,
    },
    ...ACP_PROVIDERS.map<AgentOption>((provider) => ({
      id: provider.key as OnboardingAgentId,
      label: provider.display_name,
      descriptionKey: provider.description_key,
    })),
  ];
}

interface ChooseAgentStepProps {
  selectedAgentId: OnboardingAgentId;
  onSelect: (agentId: OnboardingAgentId) => void;
  onBack?: () => void;
  onNext: () => void;
}

export function ChooseAgentStep({
  selectedAgentId,
  onSelect,
  onBack,
  onNext,
}: ChooseAgentStepProps) {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending: isSaving } = useSaveSettings();

  const handleNext = () => {
    // The diff builder seeds the preferred default model (Vertex-safe for
    // Gemini) when none is passed.
    const diff = buildAcpAgentSettingsDiff(selectedAgentId);
    if (!diff) {
      // Unknown id (shouldn't be reachable through the UI). Advance
      // without writing — better to show the next step than block the
      // user behind a silent no-op.
      onNext();
      return;
    }

    saveSettings(
      { agent_settings_diff: diff },
      {
        onError: (error) => {
          const message = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
        },
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
          onNext();
        },
      },
    );
  };

  return (
    <div
      data-testid="onboarding-step-choose-agent"
      className="flex flex-col gap-6"
    >
      <header className="flex flex-col gap-2">
        <h2 className="text-2xl font-medium text-white">
          {t(I18nKey.ONBOARDING$AGENT_TITLE)}
        </h2>
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.ONBOARDING$AGENT_SUBTITLE)}
        </p>
      </header>

      <div
        role="radiogroup"
        aria-label={t(I18nKey.ONBOARDING$AGENT_TITLE)}
        className="flex flex-col gap-3"
      >
        {getAgentOptions().map((option) => {
          const isSelected = option.id === selectedAgentId;
          return (
            <button
              key={option.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              data-testid={`onboarding-agent-option-${option.id}`}
              data-selected={isSelected}
              onClick={() => onSelect(option.id)}
              className={cn(
                "flex items-start justify-between gap-4 rounded-xl border px-4 py-3 text-left transition-colors cursor-pointer",
                isSelected
                  ? "border-white/45 bg-white/[0.09] shadow-none hover:border-white/45 hover:bg-white/[0.09]"
                  : "border-white/30 bg-white/5 hover:border-white/40 hover:bg-white/[0.08]",
              )}
            >
              <div className="flex min-w-0 flex-1 flex-col gap-1">
                <div className="flex min-w-0 items-center gap-2">
                  <AgentOptionIcon id={option.id} muted={false} />
                  <span className="truncate text-base font-normal text-white">
                    {option.label}
                  </span>
                </div>
                <span className="text-xs text-[var(--oh-muted)]">
                  {t(option.descriptionKey)}
                </span>
              </div>
              <div className="flex shrink-0 flex-col items-end gap-1">
                {isSelected ? (
                  <Check
                    width={18}
                    height={18}
                    className="mt-1 shrink-0 text-white"
                    aria-hidden
                  />
                ) : null}
              </div>
            </button>
          );
        })}
      </div>

      <div
        className={cn(
          "sticky bottom-0 flex items-center gap-2 bg-base-secondary pt-4 pb-7",
          onBack ? "justify-between" : "justify-end",
        )}
      >
        {onBack ? (
          <BrandButton
            testId="onboarding-agent-back"
            type="button"
            variant="secondary"
            onClick={onBack}
            isDisabled={isSaving}
          >
            {t(I18nKey.ONBOARDING$BACK)}
          </BrandButton>
        ) : null}
        <BrandButton
          testId="onboarding-agent-next"
          type="button"
          variant="primary"
          isDisabled={isSaving}
          onClick={handleNext}
        >
          {isSaving ? t(I18nKey.SETTINGS$SAVING) : t(I18nKey.ONBOARDING$NEXT)}
        </BrandButton>
      </div>
    </div>
  );
}
