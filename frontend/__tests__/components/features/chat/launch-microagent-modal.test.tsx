import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LaunchMicroagentModal } from "#/components/features/chat/microagent/launch-microagent-modal";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-router", async () => ({
  useParams: vi.fn().mockReturnValue({
    conversationId: "123",
  }),
}));

// Mock the useHandleRuntimeActive hook
vi.mock("#/hooks/use-handle-runtime-active", () => ({
  useHandleRuntimeActive: vi.fn().mockReturnValue({ runtimeActive: true }),
}));

// Mock the useMicroagentPrompt hook
vi.mock("#/hooks/query/use-microagent-prompt", () => ({
  useMicroagentPrompt: vi.fn().mockReturnValue({
    data: "Generated prompt",
    isLoading: false,
  }),
}));

// Mock the useGetMicroagents hook
vi.mock("#/hooks/query/use-get-microagents", () => ({
  useGetMicroagents: vi.fn().mockReturnValue({
    data: ["file1", "file2"],
  }),
}));

// Mock the useTranslation hook
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        [I18nKey.MICROAGENT$ADD_TO_MICROAGENT]: "Add to Microagent",
        [I18nKey.MICROAGENT$WHAT_TO_REMEMBER]:
          "What would you like your microagent to remember?",
        [I18nKey.MICROAGENT$WHERE_TO_PUT]: "Where should we put it?",
        [I18nKey.MICROAGENT$ADD_TRIGGERS]: "Add triggers for the microagent",
        [I18nKey.MICROAGENT$DESCRIBE_WHAT_TO_ADD]:
          "Describe what you want to add to the Microagent...",
        [I18nKey.MICROAGENT$SELECT_FILE_OR_CUSTOM]:
          "Select a microagent file or enter a custom value",
        [I18nKey.MICROAGENT$TYPE_TRIGGER_SPACE]:
          "Type a trigger and press Space to add it",
        [I18nKey.MICROAGENT$LOADING_PROMPT]: "Loading prompt...",
        [I18nKey.MICROAGENT$CANCEL]: "Cancel",
        [I18nKey.MICROAGENT$LAUNCH]: "Launch",
      };
      return translations[key] || key;
    },
    i18n: {
      changeLanguage: vi.fn(),
    },
  }),
  Trans: ({ i18nKey }: { i18nKey: string }) => i18nKey,
}));

describe("LaunchMicroagentModal", () => {
  const onCloseMock = vi.fn();
  const onLaunchMock = vi.fn();
  const eventId = 12;
  const conversationId = "123";

  const renderMicroagentModal = (
    { isLoading }: { isLoading: boolean } = { isLoading: false },
  ) =>
    render(
      <LaunchMicroagentModal
        onClose={onCloseMock}
        onLaunch={onLaunchMock}
        eventId={eventId}
        selectedRepo="some-repo"
        isLoading={isLoading}
      />,
      {
        wrapper: ({ children }) => (
          <QueryClientProvider client={new QueryClient()}>
            {children}
          </QueryClientProvider>
        ),
      },
    );

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the launch microagent modal", () => {
    renderMicroagentModal();
    expect(screen.getByTestId("launch-microagent-modal")).toBeInTheDocument();
  });

  it("should render the form fields", () => {
    renderMicroagentModal();

    // inputs
    screen.getByTestId("query-input");
    screen.getByTestId("target-input");
    screen.getByTestId("trigger-input");

    // action buttons
    screen.getByRole("button", { name: "Launch" });
    screen.getByRole("button", { name: "Cancel" });
  });

  it("should call onClose when pressing the cancel button", async () => {
    renderMicroagentModal();

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await userEvent.click(cancelButton);
    expect(onCloseMock).toHaveBeenCalled();
  });

  it("should display the prompt from the hook", async () => {
    renderMicroagentModal();

    // Since we're mocking the hook, we just need to verify the UI shows the data
    const descriptionInput = screen.getByTestId("query-input");
    expect(descriptionInput).toHaveValue("Generated prompt");
  });

  it("should display the list of microagent files from the hook", async () => {
    renderMicroagentModal();

    // Since we're mocking the hook, we just need to verify the UI shows the data
    const targetInput = screen.getByTestId("target-input");
    expect(targetInput).toHaveValue("");

    await userEvent.click(targetInput);

    expect(screen.getByText("file1")).toBeInTheDocument();
    expect(screen.getByText("file2")).toBeInTheDocument();

    await userEvent.click(screen.getByText("file1"));
    expect(targetInput).toHaveValue("file1");
  });

  it("should call onLaunch with the form data", async () => {
    renderMicroagentModal();

    const triggerInput = screen.getByTestId("trigger-input");
    await userEvent.type(triggerInput, "trigger1 ");
    await userEvent.type(triggerInput, "trigger2 ");

    const targetInput = screen.getByTestId("target-input");
    await userEvent.click(targetInput);
    await userEvent.click(screen.getByText("file1"));

    const launchButton = await screen.findByRole("button", { name: "Launch" });
    await userEvent.click(launchButton);

    expect(onLaunchMock).toHaveBeenCalledWith("Generated prompt", "file1", [
      "trigger1",
      "trigger2",
    ]);
  });

  it("should disable the launch button if isLoading is true", async () => {
    renderMicroagentModal({ isLoading: true });

    const launchButton = screen.getByRole("button", { name: "Launch" });
    expect(launchButton).toBeDisabled();
  });
});
