import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import ChevronDownIcon from "#/icons/chevron-down.svg?react";
import MessageSquareShareIcon from "#/icons/message-square-share.svg?react";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useLaunchSkillInChat } from "#/hooks/use-launch-skill-in-chat";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTracking } from "#/hooks/use-tracking";
import { getAutomationsDocsUrl } from "#/manifests/automation-interface";

const DOCS_URL = getAutomationsDocsUrl();

interface CreateInstructionsProps {
  /** If true, the instructions are collapsible and start collapsed */
  collapsible?: boolean;
}

interface CreateInstructionsContentProps {
  onLaunch?: () => void;
}

interface AutomationStartOptionProps {
  title: string;
  description: string;
  buttonLabel: string;
  testId: string;
  buttonTestId: string;
  onClick: () => void;
}

function AutomationStartOption({
  title,
  description,
  buttonLabel,
  testId,
  buttonTestId,
  onClick,
}: AutomationStartOptionProps) {
  return (
    <section
      data-testid={testId}
      className="flex h-full min-w-0 flex-col justify-between gap-4 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-4"
    >
      <div className="min-w-0">
        <h4 className="text-sm font-medium text-content">{title}</h4>
        <p className="mt-1 text-xs leading-relaxed text-tertiary-light">
          {description}
        </p>
      </div>
      <BrandButton
        type="button"
        variant="primary"
        testId={buttonTestId}
        className="mt-auto h-auto min-h-10 w-full px-4 py-2 text-center leading-tight"
        onClick={onClick}
        startContent={
          <MessageSquareShareIcon className="size-4 shrink-0" aria-hidden />
        }
      >
        {buttonLabel}
      </BrandButton>
    </section>
  );
}

export function CreateInstructionsContent({
  onLaunch,
}: CreateInstructionsContentProps = {}) {
  const { t } = useTranslation("openhands");
  const launchInChat = useLaunchSkillInChat();
  const active = useActiveBackend();
  const { trackAutomationCreatedButton } = useTracking();

  const launchAutomationPrompt = (prompt: string) => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    launchInChat(prompt, onLaunch);
  };

  const handleFindOpportunities = () => {
    launchAutomationPrompt(t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT));
  };

  const handleAddAutomation = () => {
    launchAutomationPrompt(t(I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT));
  };

  return (
    <div className="flex flex-col gap-5">
      <div className="grid gap-3 sm:grid-cols-2">
        <AutomationStartOption
          title={t(I18nKey.AUTOMATIONS$DISCOVERY_OPTION_TITLE)}
          description={t(I18nKey.AUTOMATIONS$CREATE_INSTRUCTIONS_GUIDANCE)}
          buttonLabel={t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
          testId="automations-discovery-option"
          buttonTestId="automations-find-opportunities"
          onClick={handleFindOpportunities}
        />
        <AutomationStartOption
          title={t(I18nKey.AUTOMATIONS$CUSTOM_OPTION_TITLE)}
          description={t(I18nKey.AUTOMATIONS$CUSTOM_OPTION_DESC)}
          buttonLabel={t(I18nKey.AUTOMATIONS$ADD_AUTOMATION)}
          testId="automations-add-option"
          buttonTestId="automations-add-known-automation"
          onClick={handleAddAutomation}
        />
      </div>

      <a
        href={DOCS_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-muted underline transition-colors hover:text-foreground"
      >
        {t(I18nKey.AUTOMATIONS$EMPTY_LEARN_MORE)}
      </a>
    </div>
  );
}

export function CreateInstructions({
  collapsible = false,
}: CreateInstructionsProps) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = useState(!collapsible);

  if (collapsible) {
    return (
      <div className="w-full rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)]">
        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-expanded={isExpanded}
          className="flex w-full items-center justify-between rounded-lg p-4 text-left transition-colors hover:bg-surface-raised"
        >
          <span className="text-sm font-normal text-content">
            {t(I18nKey.AUTOMATIONS$EMPTY_HOW_TO_CREATE_TITLE)}
          </span>
          <ChevronDownIcon
            className={cn(
              "size-5 text-muted transition-transform",
              isExpanded && "rotate-180",
            )}
          />
        </button>
        {isExpanded ? (
          <div className="px-4 pb-4">
            <CreateInstructionsContent />
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="w-full max-w-2xl">
      <h3 className="text-center text-sm font-medium text-content">
        {t(I18nKey.AUTOMATIONS$EMPTY_HOW_TO_CREATE_TITLE)}
      </h3>
      <div className="mt-4">
        <CreateInstructionsContent />
      </div>
    </div>
  );
}
