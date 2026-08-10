import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { CreateInstructions } from "#/components/features/automations/create-instructions";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import * as telemetry from "#/services/telemetry";

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { user_consents_to_analytics: true } }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        [I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON]:
          "Find automation opportunities",
        [I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT]:
          "Help me figure out what I should automate.",
        [I18nKey.AUTOMATIONS$CREATE_INSTRUCTIONS_GUIDANCE]:
          "OpenHands can help identify high-value automations from your recurring work.",
        [I18nKey.AUTOMATIONS$DISCOVERY_OPTION_TITLE]: "Need ideas?",
        [I18nKey.AUTOMATIONS$CUSTOM_OPTION_TITLE]: "Already know the workflow?",
        [I18nKey.AUTOMATIONS$CUSTOM_OPTION_DESC]:
          "Describe what should happen, when it should run, and where results should go.",
        [I18nKey.AUTOMATIONS$ADD_AUTOMATION]: "Add automation",
        [I18nKey.AUTOMATIONS$ADD_AUTOMATION_PROMPT]:
          "Help me add an automation.",
      };
      return translations[key] || key;
    },
  }),
  Trans: ({
    i18nKey,
    components,
  }: {
    i18nKey: string;
    components?: Record<string, React.ReactElement>;
  }) => {
    if (i18nKey !== I18nKey.AUTOMATIONS$EMPTY_OPTION_CONVERSATION_DESC) {
      return i18nKey;
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
                      "help me figure out what I should automate",
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

function renderCreateInstructions() {
  const value: NavigationContextValue = {
    currentPath: "/automations",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };

  const result = render(
    <NavigationProvider value={value}>
      <CreateInstructions />
    </NavigationProvider>,
  );

  return { ...result, navigate: value.navigate };
}

describe("CreateInstructions", () => {
  let captureMock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    captureMock = vi
      .spyOn(telemetry, "trackEvent")
      .mockResolvedValue(undefined);
    useConversationStore.setState({ messageToSend: null });
  });

  afterEach(() => {
    captureMock.mockRestore();
  });

  it("renders separate discovery and custom automation paths", () => {
    renderCreateInstructions();

    expect(
      screen.getByTestId("automations-discovery-option"),
    ).toHaveTextContent("Need ideas?");
    expect(screen.getByTestId("automations-add-option")).toHaveTextContent(
      "Already know the workflow?",
    );
    expect(
      screen.getByTestId("automations-find-opportunities"),
    ).toHaveTextContent("Find automation opportunities");
    expect(
      screen.getByTestId("automations-add-known-automation"),
    ).toHaveTextContent("Add automation");
  });

  it("captures automation_created_button with the active backend kind when a CTA is clicked", async () => {
    const user = userEvent.setup();
    renderCreateInstructions();

    await user.click(screen.getByTestId("automations-find-opportunities"));

    expect(captureMock).toHaveBeenCalledWith(
      "automation_created_button",
      expect.objectContaining({ backend_kind: "local" }),
    );
  });

  it("navigates to conversations with a discovery prompt when the discovery CTA is clicked", async () => {
    const user = userEvent.setup();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });
    const { navigate } = renderCreateInstructions();

    await user.click(screen.getByTestId("automations-find-opportunities"));

    expect(navigate).toHaveBeenCalledWith("/conversations");
    await waitFor(() => {
      expect(setMessageToSend).toHaveBeenCalledWith(
        "Help me figure out what I should automate.",
      );
    });
  });

  it("navigates to conversations with an add automation prompt when the known-workflow CTA is clicked", async () => {
    const user = userEvent.setup();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });
    const { navigate } = renderCreateInstructions();

    await user.click(screen.getByTestId("automations-add-known-automation"));

    expect(navigate).toHaveBeenCalledWith("/conversations");
    await waitFor(() => {
      expect(setMessageToSend).toHaveBeenCalledWith(
        "Help me add an automation.",
      );
    });
  });
});
