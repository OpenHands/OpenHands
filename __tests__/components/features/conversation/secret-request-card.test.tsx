import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsService } from "#/api/secrets-service";
import {
  SecretRequestCard,
  usePendingSecretRequests,
} from "#/components/features/conversation/secret-request-card";
import {
  type SecretRequest,
  useSecretRequestStore,
} from "#/stores/secret-request-store";
import { renderWithProviders } from "../../../../test-utils";

const { mockCompleteSecretRequest } = vi.hoisted(() => ({
  mockCompleteSecretRequest: vi.fn(),
}));

vi.mock("#/services/secret-request", () => ({
  completeSecretRequest: mockCompleteSecretRequest,
}));

// renderWithProviders navigates to this conversation by default.
const request: SecretRequest = {
  conversationId: "test-conversation-id",
  toolCallId: "tool-call-1",
  name: "GITHUB_TOKEN",
  reason: "Push the factory branch; needs repo scope.",
};

function PendingSecretRequestCards() {
  return usePendingSecretRequests().map((r) => (
    <SecretRequestCard key={r.toolCallId} request={r} />
  ));
}

const renderCards = async (requests: SecretRequest[] = [request]) => {
  useSecretRequestStore.setState({ requests });
  renderWithProviders(<PendingSecretRequestCards />);
  if (requests.some((r) => r.conversationId === request.conversationId)) {
    await screen.findAllByTestId("secret-request-card");
  }
};

describe("SecretRequestCard", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    // Like the real service, completing a request removes it from the queue.
    mockCompleteSecretRequest
      .mockReset()
      .mockImplementation((r: SecretRequest) =>
        useSecretRequestStore.getState().dismiss(r.toolCallId),
      );
    vi.spyOn(SecretsService, "getSecrets").mockResolvedValue([]);
  });

  it("asks for the requested secret with the agent's reason", async () => {
    await renderCards();

    expect(screen.getByText(request.reason!)).toBeInTheDocument();
    expect(screen.getByTestId("name-input")).toHaveValue("GITHUB_TOKEN");
    expect(screen.getByTestId("value-input")).toHaveValue("");
  });

  it("shows one card per pending request", async () => {
    await renderCards([
      request,
      { ...request, toolCallId: "tool-call-2", name: "NPM_TOKEN" },
    ]);

    const names = screen
      .getAllByTestId("name-input")
      .map((input) => (input as HTMLInputElement).value);
    expect(names).toEqual(["GITHUB_TOKEN", "NPM_TOKEN"]);
  });

  it("ignores requests from other conversations", async () => {
    await renderCards([{ ...request, conversationId: "other-conversation" }]);

    expect(screen.queryByTestId("secret-request-card")).not.toBeInTheDocument();
  });

  it("saves the value to the secrets store and completes with the secret's identity", async () => {
    const user = userEvent.setup();
    const createSecret = vi
      .spyOn(SecretsService, "createSecret")
      .mockResolvedValue(undefined);
    await renderCards();

    const valueInput = screen.getByTestId("value-input");
    expect(valueInput).toHaveAttribute("type", "password");
    await user.type(valueInput, "ghp_example");
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() =>
      expect(
        screen.queryByTestId("secret-request-card"),
      ).not.toBeInTheDocument(),
    );
    // The form's later onCancel is a no-op for the real, already-completed
    // request (covered in secret-request.test.ts).
    expect(mockCompleteSecretRequest).toHaveBeenNthCalledWith(1, request, {
      name: "GITHUB_TOKEN",
      description: undefined,
    });
    expect(createSecret).toHaveBeenCalledWith(
      "GITHUB_TOKEN",
      "ghp_example",
      undefined,
    );
  });

  it("stays after a failed save so the user can retry or cancel", async () => {
    const user = userEvent.setup();
    vi.spyOn(SecretsService, "createSecret").mockRejectedValue(
      new Error("Save failed"),
    );
    await renderCards();

    await user.type(screen.getByTestId("value-input"), "ghp_example");
    await user.click(screen.getByTestId("submit-button"));

    await waitFor(() =>
      expect(screen.getByTestId("cancel-button")).toBeEnabled(),
    );
    expect(screen.getByTestId("secret-request-card")).toBeInTheDocument();
    expect(mockCompleteSecretRequest).not.toHaveBeenCalled();

    await user.click(screen.getByTestId("cancel-button"));

    expect(mockCompleteSecretRequest).toHaveBeenCalledExactlyOnceWith(
      request,
      null,
    );
  });
});
