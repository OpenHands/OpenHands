import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { AddAutomationModal } from "#/components/features/automations/add-automation-modal";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { user_consents_to_analytics: true } }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
  Trans: ({
    i18nKey,
    components,
    children,
  }: {
    i18nKey: string;
    components?: Record<string, React.ReactElement>;
    children?: React.ReactNode;
  }) => {
    if (i18nKey !== I18nKey.AUTOMATIONS$EMPTY_OPTION_CONVERSATION_DESC) {
      return children ?? i18nKey;
    }

    return (
      <>
        Start a new conversation and tell OpenHands to{" "}
        {components?.example
          ? React.cloneElement(
              components.example,
              {},
              <>
                {components.cmd
                  ? React.cloneElement(
                      components.cmd,
                      {},
                      "Create an automation",
                    )
                  : null}
                {components.punct
                  ? React.cloneElement(components.punct, {}, ".")
                  : null}
              </>,
            )
          : null}
      </>
    );
  },
}));

function renderModal(isOpen = true) {
  const onClose = vi.fn();
  const navigation: NavigationContextValue = {
    currentPath: "/automations",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };

  render(
    <NavigationProvider value={navigation}>
      <AddAutomationModal isOpen={isOpen} onClose={onClose} />
    </NavigationProvider>,
  );

  return { onClose };
}

describe("AddAutomationModal", () => {
  it("renders the create instructions content when open", () => {
    renderModal();

    expect(screen.getByTestId("add-automation-modal")).toBeInTheDocument();
    expect(
      screen.getByTestId("automations-create-instructions-example"),
    ).toHaveTextContent("Create an automation");
    expect(
      screen.getByTestId("automations-create-automation"),
    ).toHaveTextContent(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON);
    expect(
      screen.queryByText(I18nKey.AUTOMATIONS$EMPTY_OPTION_PLUGIN_TITLE),
    ).not.toBeInTheDocument();
  });

  it("does not render when closed", () => {
    renderModal(false);

    expect(screen.queryByTestId("add-automation-modal")).not.toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const { onClose } = renderModal();

    await user.click(screen.getByTestId("add-automation-modal-close"));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
