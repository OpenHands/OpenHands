import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NewProjectButton } from "#/components/shared/buttons/new-project-button";

const mutateMock = vi.fn((_: object, options?: { onSuccess?: (data: { conversation_id: string }) => void }) => {
  options?.onSuccess?.({ conversation_id: "new-conv-id" });
});

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        if (key === "CONVERSATION$START_NEW") return "Start new conversation";
        return key;
      },
      i18n: { language: "en" },
    }),
  };
});

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutate: mutateMock,
    isPending: false,
    isSuccess: false,
  }),
}));

vi.mock("#/hooks/use-is-creating-conversation", () => ({
  useIsCreatingConversation: () => false,
}));

const renderButton = () => {
  const RouterStub = createRoutesStub([
    {
      path: "/conversation/:conversationId",
      Component: () => <NewProjectButton />,
    },
    {
      path: "/conversations/:conversationId",
      Component: () => <div data-testid="conversation-screen" />,
    },
  ]);

  return render(<RouterStub initialEntries={["/conversation/123"]} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
};

describe("NewProjectButton", () => {
  beforeEach(() => {
    mutateMock.mockClear();
  });

  it("creates and navigates to a new conversation when clicked", async () => {
    renderButton();

    await userEvent.click(screen.getByTestId("new-project-button"));

    expect(mutateMock).toHaveBeenCalledTimes(1);
    await screen.findByTestId("conversation-screen");
  });
});
