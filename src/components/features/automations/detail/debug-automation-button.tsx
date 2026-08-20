import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useNavigation } from "#/context/navigation-context";
import { buildAutomationDebugPrompt } from "#/utils/automation-debug-prompt";
import type { Automation, AutomationRun } from "#/types/automation";

interface DebugAutomationButtonProps {
  /** The failed run being debugged. */
  run: AutomationRun;
  /** The parent automation, used to add context to the debug prompt. */
  automation?: Automation;
  /** The run's stderr, already computed by the enclosing logs modal. */
  stderr: string;
}

/**
 * Starts a new OpenHands conversation seeded with the failed run's error
 * details so the agent begins debugging immediately. Kept as its own component
 * (rather than inlined into the logs modal) so `useCreateConversation` — which
 * requires a QueryClientProvider — only mounts for failed runs.
 */
export function DebugAutomationButton({
  run,
  automation,
  stderr,
}: DebugAutomationButtonProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const createConversation = useCreateConversation();

  const handleClick = () => {
    if (createConversation.isPending) return;

    const query = buildAutomationDebugPrompt({
      automationName: automation?.name,
      automationPrompt: automation?.prompt,
      errorDetail: run.error_detail,
      stderr,
      runId: run.id,
    });

    createConversation.mutate(
      { query },
      {
        onSuccess: (data) => {
          navigate(`/conversations/${data.conversation_id}`);
        },
      },
    );
  };

  return (
    <BrandButton
      testId="debug-automation-button"
      type="button"
      variant="primary"
      onClick={handleClick}
      isDisabled={createConversation.isPending}
      aria-busy={createConversation.isPending}
    >
      {t(I18nKey.AUTOMATIONS$DETAIL$DEBUG_WITH_OPENHANDS)}
    </BrandButton>
  );
}
