import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithProviders } from "test-utils";
import { StartTaskStatusBadge } from "#/components/features/conversation-panel/start-task-card/start-task-status-badge";
import { I18nKey } from "#/i18n/declaration";

describe("StartTaskStatusBadge", () => {
  // react-i18next is globally mocked so `t` echoes the key; asserting the key
  // proves the label is resolved through i18n rather than a raw enum transform.
  it.each([
    ["STARTING_CONVERSATION", I18nKey.CONVERSATION$STARTING_CONVERSATION],
    ["READY", I18nKey.CONVERSATION$READY],
    ["ERROR", I18nKey.COMMON$ERROR],
  ] as const)(
    "localizes the label for the %s status",
    (taskStatus, expectedKey) => {
      renderWithProviders(<StartTaskStatusBadge taskStatus={taskStatus} />);

      expect(screen.getByText(expectedKey)).toBeInTheDocument();
    },
  );

  it("renders nothing for the WORKING status", () => {
    const { container } = renderWithProviders(
      <StartTaskStatusBadge taskStatus="WORKING" />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
