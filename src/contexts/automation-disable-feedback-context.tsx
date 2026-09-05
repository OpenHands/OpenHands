import React from "react";
import type { AutomationDisableAnalyticsContext } from "#/types/automation-disable-feedback";

interface AutomationDisableFeedbackContextValue {
  requestAutomationDisableFeedback: (
    context: AutomationDisableAnalyticsContext,
  ) => void;
}

const AutomationDisableFeedbackContext = React.createContext<
  AutomationDisableFeedbackContextValue | undefined
>(undefined);

export const AutomationDisableFeedbackContextProvider =
  AutomationDisableFeedbackContext.Provider;

/**
 * A no-op fallback keeps mutation hooks usable in isolated component tests.
 * Production entry points install AutomationDisableFeedbackProvider.
 */
export function useAutomationDisableFeedback() {
  return (
    React.useContext(AutomationDisableFeedbackContext) ?? {
      requestAutomationDisableFeedback: () => undefined,
    }
  );
}
