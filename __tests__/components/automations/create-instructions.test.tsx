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

const automationConversationMocks = vi.hoisted(() => ({
  createConversationMutate: vi.fn(),
  isCreatingConversation: vi.fn(() => false),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { user_consents_to_analytics: true } }),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: automationConversationMocks.createConversationMutate,
    isPending: false,
  }),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () =>
    automationConversationMocks.isCreatingConversation(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        [I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON]: "Find opportunities",
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
        [I18nKey.HOME$CREATING_CONVERSATION]: "Creating conversation…",
        [I18nKey.ERROR$GENERIC]: "Something went wrong.",
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

    const chip = (text: string) =>
      components?.cmd ? React.cloneElement(components.cmd, {}, text) : text;

    return (
      <>
        Start a conversation and ask OpenHands to{" "}
        {components?.example
          ? React.cloneElement(
              components.example,
              {},
              chip("help me figure out what I should automate"),
            )
          : null}{" "}
        or{" "}
        {components?.example
          ? React.cloneElement(
              components.example,
              {},
              <>
                {chip("Create an automation")}
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
    automationConversationMocks.isCreatingConversation.mockReturnValue(false);
    automationConversationMocks.createConversationMutate.mockImplementation(
      (_variables, options) => {
        options?.onSuccess?.({
          conversation_id: "automation-conversation",
          session_api_key: null,
          url: null,
        });
      },
    );
    useConversationStore.setState({ messageToSend: null });
  });

  afterEach(() => {
    captureMock.mockRestore();
    automationConversationMocks.createConversationMutate.mockReset();
    automationConversationMocks.isCreatingConversation.mockReset();
    window.localStorage.clear();
  });

  it("renders add and find-opportunities actions", () => {
    renderCreateInstructions();

    expect(
      screen.queryByTestId("automations-discovery-option"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automations-add-known-automation"),
    ).toHaveTextContent("Add automation");
    expect(
      screen.getByTestId("automations-find-opportunities"),
    ).toHaveTextContent("Find opportunities");
    expect(
      screen.getAllByTestId("automations-create-instructions-example").map(
        (element) => element.textContent,
      ),
    ).toEqual([
      "help me figure out what I should automate",
      "Create an automation",
    ]);
  });

  it("captures automation_created_button with intent and source when a CTA is clicked", async () => {
    const user = userEvent.setup();
    renderCreateInstructions();

    await user.click(screen.getByTestId("automations-find-opportunities"));

    expect(captureMock).toHaveBeenCalledWith(
      "automation_created_button",
      expect.objectContaining({
        backend_kind: "local",
        intent: "find_opportunities",
        source: "empty_state",
      }),
    );
  });

  it("shows a launch modal while creating the seeded conversation", async () => {
    const user = userEvent.setup();
    automationConversationMocks.createConversationMutate.mockImplementationOnce(
      () => {},
    );
    renderCreateInstructions();

    await user.click(screen.getByTestId("automations-find-opportunities"));

    expect(
      await screen.findByTestId("automation-conversation-launch-modal"),
    ).toHaveTextContent("Creating conversation…");
  });

  it("creates a conversation with a discovery prompt when the discovery CTA is clicked", async () => {
    const user = userEvent.setup();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });
    const { navigate } = renderCreateInstructions();

    await user.click(screen.getByTestId("automations-find-opportunities"));

    expect(
      automationConversationMocks.createConversationMutate,
    ).toHaveBeenCalledWith(
      {},
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    expect(navigate).toHaveBeenCalledWith(
      "/conversations/automation-conversation",
    );
    expect(
      JSON.parse(
        window.localStorage.getItem(
          "conversation-state-automation-conversation",
        ) ?? "{}",
      ).draftMessage,
    ).toBe("Help me figure out what I should automate.");
    await waitFor(() => {
      expect(setMessageToSend).toHaveBeenCalledWith(
        "Help me figure out what I should automate.",
      );
    });
  });

  it("creates a conversation with an add automation prompt when the known-workflow CTA is clicked", async () => {
    const user = userEvent.setup();
    const setMessageToSend = vi.fn();
    useConversationStore.setState({ setMessageToSend });
    const { navigate } = renderCreateInstructions();

    await user.click(screen.getByTestId("automations-add-known-automation"));

    expect(navigate).toHaveBeenCalledWith(
      "/conversations/automation-conversation",
    );
    expect(
      JSON.parse(
        window.localStorage.getItem(
          "conversation-state-automation-conversation",
        ) ?? "{}",
      ).draftMessage,
    ).toBe("Help me add an automation.");
    await waitFor(() => {
      expect(setMessageToSend).toHaveBeenCalledWith(
        "Help me add an automation.",
      );
    });
  });
});
