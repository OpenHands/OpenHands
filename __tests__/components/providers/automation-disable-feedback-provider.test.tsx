/* eslint-disable i18next/no-literal-string -- test-only labels and host content */
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock(
  "#/components/features/automations/automation-disable-feedback-prompt",
  () => {
    throw new Error("feedback chunk failed to load");
  },
);

import { AutomationDisableFeedbackProvider } from "#/components/providers/automation-disable-feedback-provider";
import { useAutomationDisableFeedback } from "#/contexts/automation-disable-feedback-context";

function FeedbackRequester() {
  const { requestAutomationDisableFeedback } = useAutomationDisableFeedback();

  return (
    <button
      type="button"
      onClick={() =>
        requestAutomationDisableFeedback({
          backendKind: "local",
          automationId: "automation-1",
          automationType: "schedule",
          disablementId: "disablement-1",
        })
      }
    >
      Request feedback
    </button>
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AutomationDisableFeedbackProvider", () => {
  it("keeps host content mounted when the optional prompt fails to load", async () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);

    render(
      <AutomationDisableFeedbackProvider>
        <div data-testid="host-content">Host content</div>
        <FeedbackRequester />
      </AutomationDisableFeedbackProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Request feedback" }));

    await waitFor(() => expect(consoleError).toHaveBeenCalled());
    expect(screen.getByTestId("host-content")).toHaveTextContent(
      "Host content",
    );
    expect(
      screen.getByRole("button", { name: "Request feedback" }),
    ).toBeEnabled();
  });
});
