import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import CopyIcon from "#/icons/copy.svg?react";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import MessageSquareShareIcon from "#/icons/message-square-share.svg?react";
import { SkillIconBadge } from "./skill-icon-badge";
import { getSkillCardDescription } from "./get-skill-card-description";
import { buildSkillPills } from "./build-skill-pills";
import { isCopyableSkillSource } from "./is-copyable-skill-source";
import { SkillCardPillRow } from "./skill-card-pill-row";
import { getSkillChatLaunchMessage } from "./get-skill-chat-launch-message";
import { useLaunchSkillInChat } from "#/hooks/use-launch-skill-in-chat";

interface SkillDetailModalProps {
  skill: SkillInfo;
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
        rows={Math.min(12, Math.max(4, value.split("\n").length))}
        className={cn(
          "bg-[var(--oh-surface-raised)] border border-[var(--oh-border-subtle)] w-full min-w-0 rounded-sm p-2 text-sm",
          "cursor-not-allowed resize-none custom-scrollbar",
        )}
      />
    </label>
  );
}

export function SkillDetailModal({
  skill,
  enabled,
  onToggle,
  onClose,
}: SkillDetailModalProps) {
  const { t } = useTranslation("openhands");
  const launchSkillInChat = useLaunchSkillInChat();
  const [sourceCopied, setSourceCopied] = React.useState(false);
  const chatLaunchMessage = React.useMemo(
    () => getSkillChatLaunchMessage(skill),
    [skill],
  );

  const description = getSkillCardDescription(skill);
  const pills = React.useMemo(
    () =>
      buildSkillPills(skill, t, {
        variant: "detail",
        testIdPrefix: "skill-modal-pill",
      }),
    [skill, t],
  );
  const showCopySource = isCopyableSkillSource(skill.source);

  const handleCopySource = async () => {
    if (!skill.source) {
      return;
    }

    await navigator.clipboard.writeText(skill.source);
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
    <ModalBackdrop onClose={onClose} aria-label={skill.name}>
      <div
        data-testid="skill-detail-modal"
        data-skill-name={skill.name}
        className="relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-[var(--oh-border)] w-[520px] max-w-[90vw] max-h-[85vh] overflow-y-auto custom-scrollbar"
      >
        <ModalCloseButton onClose={onClose} testId="skill-detail-modal-close" />
        <div className="flex items-start gap-3 pr-6">
          <SkillIconBadge skillName={skill.name} />
          <div className="min-w-0 flex-1">
            <h2
              data-testid={`skill-modal-name-${skill.name}`}
              className={modalTitleLgClassName}
            >
              {skill.name}
            </h2>
            {skill.source ? (
              <div className="mt-0.5 flex min-w-0 items-center gap-1">
                <p
                  data-testid={`skill-modal-source-${skill.name}`}
                  className="min-w-0 flex-1 truncate text-xs text-tertiary-alt"
                  title={skill.source}
                >
                  {skill.source}
                </p>
                {showCopySource ? (
                  <button
                    type="button"
                    data-testid={`skill-modal-copy-source-${skill.name}`}
                    aria-label={t(
                      sourceCopied
                        ? I18nKey.BUTTON$COPIED
                        : I18nKey.SETTINGS$SKILLS_COPY_PATH,
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
          data-testid={`skill-modal-enable-row-${skill.name}`}
          className="flex w-full items-center rounded-lg border border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)] px-3 py-2.5"
        >
          <SettingsSwitch
            testId={`skill-modal-toggle-${skill.name}`}
            isToggled={enabled}
            onToggle={onToggle}
            togglePosition="right"
          >
            {t(
              enabled
                ? I18nKey.SETTINGS$SKILLS_ENABLED
                : I18nKey.SETTINGS$SKILLS_DISABLED,
            )}
          </SettingsSwitch>
        </div>

        {description ? (
          <p
            data-testid={`skill-modal-description-${skill.name}`}
            className="text-xs text-tertiary-light"
          >
            {description}
          </p>
        ) : null}

        {pills.length > 0 ? (
          <SkillCardPillRow
            pills={pills}
            testId={`skill-modal-pills-${skill.name}`}
          />
        ) : null}

        {skill.content ? (
          <ReadonlyTextArea
            testId={`skill-modal-field-content-${skill.name}`}
            label={t(I18nKey.SETTINGS$SKILLS_CONTENT)}
            value={skill.content}
          />
        ) : null}

        <div className="mt-2 flex justify-end gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="skill-detail-close"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            isDisabled={!enabled}
            onClick={() => launchSkillInChat(chatLaunchMessage, onClose)}
            testId={`skill-detail-use-skill-${skill.name}`}
            startContent={
              <MessageSquareShareIcon className="size-4" aria-hidden />
            }
          >
            {t(I18nKey.SETTINGS$SKILLS_USE_SKILL_BUTTON)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
