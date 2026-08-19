import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import type { AgentInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import CopyIcon from "#/icons/copy.svg?react";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import { SkillCardPillRow } from "#/components/features/skills/skill-card-pill-row";
import { isCopyableSkillSource } from "#/components/features/skills/is-copyable-skill-source";
import { AgentIconBadge } from "./agent-icon-badge";
import { getAgentCardDescription } from "./get-agent-card-description";
import { buildAgentPills } from "./build-agent-pills";

interface AgentDetailModalProps {
  agent: AgentInfo;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  onClose: () => void;
}

function ReadonlyTextArea({
  testId,
  label,
  value,
}: {
  testId: string;
  label: string;
  value: string;
}) {
  return (
    <label className="flex min-w-0 w-full flex-col gap-2.5">
      <span className="text-sm">{label}</span>
      <textarea
        data-testid={testId}
        readOnly
        value={value}
        rows={Math.min(16, Math.max(4, value.split("\n").length))}
        className={cn(
          "bg-[var(--oh-surface-raised)] border border-[var(--oh-border-subtle)] w-full min-w-0 rounded-sm p-2 text-sm",
          "cursor-not-allowed resize-none custom-scrollbar",
        )}
      />
    </label>
  );
}

export function AgentDetailModal({
  agent,
  enabled,
  onToggle,
  onClose,
}: AgentDetailModalProps) {
  const { t } = useTranslation("openhands");
  const [sourceCopied, setSourceCopied] = React.useState(false);

  const description = getAgentCardDescription(agent);
  const pills = React.useMemo(
    () =>
      buildAgentPills(agent, {
        testIdPrefix: "agent-modal-pill",
      }),
    [agent],
  );
  const whenToUse = agent.when_to_use_examples ?? [];
  const showCopySource = isCopyableSkillSource(agent.source);

  const handleCopySource = async () => {
    if (!agent.source) {
      return;
    }

    await navigator.clipboard.writeText(agent.source);
    setSourceCopied(true);
  };

  React.useEffect(() => {
    if (!sourceCopied) {
      return undefined;
    }

    const timeout = setTimeout(() => setSourceCopied(false), 2000);
    return () => clearTimeout(timeout);
  }, [sourceCopied]);

  return (
    <ModalBackdrop onClose={onClose} aria-label={agent.name}>
      <div
        data-testid="agent-detail-modal"
        data-agent-name={agent.name}
        className="relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-[var(--oh-border)] w-[520px] max-w-[90vw] max-h-[85vh] overflow-y-auto custom-scrollbar"
      >
        <ModalCloseButton onClose={onClose} testId="agent-detail-modal-close" />
        <div className="flex items-start gap-3 pr-6">
          <AgentIconBadge agentName={agent.name} />
          <div className="min-w-0 flex-1">
            <h2
              data-testid={`agent-modal-name-${agent.name}`}
              className={modalTitleLgClassName}
            >
              {agent.name}
            </h2>
            {agent.source ? (
              <div className="mt-0.5 flex min-w-0 items-center gap-1">
                <p
                  data-testid={`agent-modal-source-${agent.name}`}
                  className="min-w-0 flex-1 truncate text-xs text-tertiary-alt"
                  title={agent.source}
                >
                  {agent.source}
                </p>
                {showCopySource ? (
                  <button
                    type="button"
                    data-testid={`agent-modal-copy-source-${agent.name}`}
                    aria-label={t(
                      sourceCopied
                        ? I18nKey.BUTTON$COPIED
                        : I18nKey.SETTINGS$AGENTS_COPY_PATH,
                    )}
                    disabled={sourceCopied}
                    onClick={handleCopySource}
                    className="shrink-0 cursor-pointer border-0 bg-transparent p-0.5 text-tertiary-alt hover:text-white disabled:cursor-default [&_path]:fill-current"
                  >
                    {sourceCopied ? (
                      <CheckmarkIcon width={12} height={12} />
                    ) : (
                      <CopyIcon width={12} height={12} />
                    )}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        <div
          data-testid={`agent-modal-enable-row-${agent.name}`}
          className="flex w-full items-center rounded-lg border border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)] px-3 py-2.5"
        >
          <SettingsSwitch
            testId={`agent-modal-toggle-${agent.name}`}
            isToggled={enabled}
            onToggle={onToggle}
            togglePosition="right"
          >
            {t(
              enabled
                ? I18nKey.SETTINGS$AGENTS_ENABLED
                : I18nKey.SETTINGS$AGENTS_DISABLED,
            )}
          </SettingsSwitch>
        </div>

        {description ? (
          <p
            data-testid={`agent-modal-description-${agent.name}`}
            className="text-xs text-tertiary-light"
          >
            {description}
          </p>
        ) : null}

        {pills.length > 0 ? (
          <SkillCardPillRow
            pills={pills}
            testId={`agent-modal-pills-${agent.name}`}
          />
        ) : null}

        {whenToUse.length > 0 ? (
          <div
            data-testid={`agent-modal-when-to-use-${agent.name}`}
            className="flex flex-col gap-1.5"
          >
            <span className="text-sm">
              {t(I18nKey.SETTINGS$AGENTS_WHEN_TO_USE)}
            </span>
            <ul className="list-disc space-y-1 pl-4 text-xs leading-relaxed text-tertiary-light">
              {whenToUse.map((example) => (
                <li key={example}>{example}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {agent.system_prompt ? (
          <ReadonlyTextArea
            testId={`agent-modal-field-system-prompt-${agent.name}`}
            label={t(I18nKey.SETTINGS$AGENTS_SYSTEM_PROMPT)}
            value={agent.system_prompt}
          />
        ) : null}

        <div className="mt-2 flex justify-end gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="agent-detail-close"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
