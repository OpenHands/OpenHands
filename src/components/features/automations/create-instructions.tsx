import { useState, type ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";
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

function InlineExampleWrap({ children }: { children?: ReactNode }) {
  return <span className="whitespace-nowrap">{children}</span>;
}

function InlineCodeChip({ children }: { children?: ReactNode }) {
  return (
    <code
      data-testid="automations-create-instructions-example"
      className={cn(
        "mx-0.5 inline-block rounded-sm border border-[var(--oh-border-subtle)]",
        "bg-[var(--oh-surface-raised)] px-1.5 py-0.5 align-baseline font-mono text-[11px] text-white",
      )}
    >
      {children}
    </code>
  );
}

function InlinePunctuation({ children }: { children?: ReactNode }) {
  return <>{children}</>;
}

const CREATE_INSTRUCTIONS_INLINE_COMPONENTS = {
  example: <InlineExampleWrap />,
  cmd: <InlineCodeChip />,
  punct: <InlinePunctuation />,
};

interface CreateInstructionsProps {
  /** If true, the instructions are collapsible and start collapsed */
  collapsible?: boolean;
}

interface CreateInstructionsContentProps {
  onLaunch?: () => void;
}

export function CreateInstructionsContent({
  onLaunch,
}: CreateInstructionsContentProps = {}) {
  const { t } = useTranslation("openhands");
  const launchInChat = useLaunchSkillInChat();
  const active = useActiveBackend();
  const { trackAutomationCreatedButton } = useTracking();

  const handleCreateAutomation = () => {
    trackAutomationCreatedButton({ backendKind: active.backend.kind });
    launchInChat(t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT), onLaunch);
  };

  return (
    <div className="flex flex-col gap-5">
      <p className="text-sm leading-relaxed text-tertiary-light">
        <Trans
          ns="openhands"
          i18nKey={I18nKey.AUTOMATIONS$EMPTY_OPTION_CONVERSATION_DESC}
          components={CREATE_INSTRUCTIONS_INLINE_COMPONENTS}
        />{" "}
        {t(I18nKey.AUTOMATIONS$CREATE_INSTRUCTIONS_GUIDANCE)}
      </p>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-muted underline transition-colors hover:text-foreground"
        >
          {t(I18nKey.AUTOMATIONS$EMPTY_LEARN_MORE)}
        </a>
        <BrandButton
          type="button"
          variant="primary"
          testId="automations-create-automation"
          onClick={handleCreateAutomation}
          startContent={
            <MessageSquareShareIcon className="size-4" aria-hidden />
          }
        >
          {t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
        </BrandButton>
      </div>
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
