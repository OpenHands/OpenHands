import { useState, type ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Plus, Sparkles } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import ChevronDownIcon from "#/icons/chevron-down.svg?react";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import { getAutomationsDocsUrl } from "#/manifests/automation-interface";
import { AutomationConversationLaunchModal } from "./automation-conversation-launch-modal";
import type { AutomationConversationLaunchRequest } from "./use-launch-automation-conversation";

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
  const [launchRequest, setLaunchRequest] =
    useState<AutomationConversationLaunchRequest | null>(null);

  const handleFindOpportunities = () => {
    setLaunchRequest({
      intent: "find_opportunities",
      source: "empty_state",
      prompt: t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT),
    });
  };

  const handleAddAutomation = () => {
    setLaunchRequest({
      intent: "add_automation",
      source: "empty_state",
      prompt: t(I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT),
    });
  };

  const handleLaunchModalClose = () => {
    setLaunchRequest(null);
    onLaunch?.();
  };

  return (
    <>
      <div className="flex flex-col gap-5">
        <p className="text-sm leading-relaxed text-tertiary-light">
          <Trans
            ns="openhands"
            i18nKey={I18nKey.AUTOMATIONS$EMPTY_OPTION_CONVERSATION_DESC}
            components={CREATE_INSTRUCTIONS_INLINE_COMPONENTS}
          />{" "}
          {t(I18nKey.AUTOMATIONS$CREATE_INSTRUCTIONS_GUIDANCE)}{" "}
          <a
            href={DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted underline transition-colors hover:text-foreground"
          >
            {t(I18nKey.AUTOMATIONS$EMPTY_LEARN_MORE)}
          </a>
        </p>

        <div className="flex flex-wrap justify-center gap-2">
          <BrandButton
            type="button"
            variant="primary"
            testId="automations-add-known-automation"
            onClick={handleAddAutomation}
            startContent={<Plus className="size-4" aria-hidden />}
          >
            {t(I18nKey.AUTOMATIONS$ADD_AUTOMATION)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            testId="automations-find-opportunities"
            onClick={handleFindOpportunities}
            startContent={<Sparkles className="size-4" aria-hidden />}
          >
            {t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
          </BrandButton>
        </div>
      </div>
      <AutomationConversationLaunchModal
        request={launchRequest}
        onClose={handleLaunchModalClose}
      />
    </>
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
      <CreateInstructionsContent />
    </div>
  );
}
