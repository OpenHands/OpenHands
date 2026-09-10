import { AutomationDisableFeedbackModal } from "#/components/features/automations/automation-disable-feedback-modal";
import { useTracking } from "#/hooks/use-tracking";
import type {
  AutomationDisableAnalyticsContext,
  AutomationDisableFeedback,
} from "#/types/automation-disable-feedback";

interface AutomationDisableFeedbackPromptProps {
  context: AutomationDisableAnalyticsContext;
  onClose: () => void;
}

export function AutomationDisableFeedbackPrompt({
  context,
  onClose,
}: AutomationDisableFeedbackPromptProps) {
  const { trackAutomationDisableFeedback } = useTracking();

  const handleSubmit = (feedback: AutomationDisableFeedback) => {
    trackAutomationDisableFeedback({ ...context, ...feedback });
    onClose();
  };

  return (
    <AutomationDisableFeedbackModal
      onSubmit={handleSubmit}
      onDismiss={onClose}
    />
  );
}
