import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SayHelloStep } from "#/components/features/onboarding/steps/say-hello-step";
import { renderWithProviders } from "../../../test-utils";

const createConversation = vi.hoisted(() => vi.fn());

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "cloud" } }),
}));

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: createConversation,
    isPending: false,
    isSuccess: false,
  }),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

describe("SayHelloStep", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("reports pending immediately, blocks duplicate sends, and navigates only on success", async () => {
    const navigate = vi.fn();
    const onLaunched = vi.fn();
    renderWithProviders(
      <SayHelloStep
        onBack={vi.fn()}
        onClose={vi.fn()}
        onLaunched={onLaunched}
      />,
      { navigation: { navigate } },
    );

    const send = screen.getByTestId("submit-button");
    await userEvent.click(send);

    expect(screen.getByRole("status")).toHaveTextContent(
      "ONBOARDING$HELLO_LAUNCHING",
    );
    expect(send).toBeDisabled();
    expect(
      screen.getByTestId("submit-button-pending-icon"),
    ).toBeInTheDocument();
    await userEvent.click(send);
    expect(createConversation).toHaveBeenCalledTimes(1);
    expect(navigate).not.toHaveBeenCalled();
    expect(onLaunched).not.toHaveBeenCalled();

    const callbacks = createConversation.mock.calls[0][1];
    callbacks.onSuccess({ conversation_id: "conversation-123" });

    expect(navigate).toHaveBeenCalledWith("/conversations/conversation-123");
    expect(onLaunched).toHaveBeenCalledTimes(1);
  });

  it("shows an actionable error and allows retry after create fails", async () => {
    renderWithProviders(
      <SayHelloStep onBack={vi.fn()} onClose={vi.fn()} onLaunched={vi.fn()} />,
    );

    await userEvent.click(screen.getByTestId("submit-button"));
    createConversation.mock.calls[0][1].onError(new Error("backend down"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("ERROR$GENERIC");
    expect(alert).toHaveTextContent("LAUNCH$TRY_AGAIN");
    expect(screen.getByTestId("submit-button")).toBeEnabled();

    await userEvent.click(
      screen.getByRole("button", { name: "LAUNCH$TRY_AGAIN" }),
    );
    expect(createConversation).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
