import React from "react";
import { AutomationDisableFeedbackContextProvider } from "#/contexts/automation-disable-feedback-context";
import type { AutomationDisableAnalyticsContext } from "#/types/automation-disable-feedback";

const AutomationDisableFeedbackPrompt = React.lazy(() =>
  import("#/components/features/automations/automation-disable-feedback-prompt").then(
    (module) => ({
      default: module.AutomationDisableFeedbackPrompt,
    }),
  ),
);

interface AutomationDisableFeedbackErrorBoundaryProps
  extends React.PropsWithChildren {
  onError: () => void;
}

class AutomationDisableFeedbackErrorBoundary extends React.Component<
  AutomationDisableFeedbackErrorBoundaryProps,
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch() {
    this.props.onError();
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function AutomationDisableFeedbackProvider({
  children,
}: React.PropsWithChildren) {
  const [pendingFeedback, setPendingFeedback] =
    React.useState<AutomationDisableAnalyticsContext | null>(null);

  const requestAutomationDisableFeedback = React.useCallback(
    (context: AutomationDisableAnalyticsContext) => {
      setPendingFeedback(context);
    },
    [],
  );

  const dismissFeedback = React.useCallback(() => {
    setPendingFeedback(null);
  }, []);

  const contextValue = React.useMemo(
    () => ({ requestAutomationDisableFeedback }),
    [requestAutomationDisableFeedback],
  );

  return (
    <AutomationDisableFeedbackContextProvider value={contextValue}>
      {children}
      {pendingFeedback ? (
        <AutomationDisableFeedbackErrorBoundary onError={dismissFeedback}>
          <React.Suspense fallback={null}>
            <AutomationDisableFeedbackPrompt
              context={pendingFeedback}
              onClose={dismissFeedback}
            />
          </React.Suspense>
        </AutomationDisableFeedbackErrorBoundary>
      ) : null}
    </AutomationDisableFeedbackContextProvider>
  );
}
