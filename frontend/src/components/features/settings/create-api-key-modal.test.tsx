import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreateApiKeyModal } from "./create-api-key-modal";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const mockState = vi.hoisted(() => ({
  mutateAsync: vi.fn(),
  invalidateQueries: vi.fn(),
  isPending: false,
}));

vi.mock("#/hooks/mutation/use-create-api-key", () => ({
  useCreateApiKey: () => ({
    mutateAsync: mockState.mutateAsync,
    isPending: mockState.isPending,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const renderModal = (props: {
  onKeyCreated?: (key: unknown) => void;
  onClose?: () => void;
}) =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <CreateApiKeyModal
        isOpen
        onClose={props.onClose ?? vi.fn()}
        onKeyCreated={props.onKeyCreated ?? vi.fn()}
      />
    </QueryClientProvider>,
  );

describe("CreateApiKeyModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState.mutateAsync.mockResolvedValue({
      id: "1",
      name: "Test",
      key: "sk-oh-test",
      prefix: "oh_1_",
      created_at: "2026-06-01T00:00:00Z",
      not_before: null,
      expires_at: null,
    });
  });

  it("renders the new active-window date inputs", () => {
    renderModal({});
    expect(screen.getByTestId("api-key-not-before-input")).toBeInTheDocument();
    expect(screen.getByTestId("api-key-expires-at-input")).toBeInTheDocument();
  });

  it("submits with name only when no dates are set", async () => {
    const onKeyCreated = vi.fn();
    renderModal({ onKeyCreated });

    fireEvent.change(screen.getByTestId("api-key-name-input"), {
      target: { value: "My Key" },
    });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CREATE" }));

    await waitFor(() => {
      expect(mockState.mutateAsync).toHaveBeenCalledWith({
        name: "My Key",
        not_before: undefined,
        expires_at: undefined,
      });
    });
    expect(onKeyCreated).toHaveBeenCalled();
  });

  it("submits with name + active window when both dates are set", async () => {
    const onKeyCreated = vi.fn();
    renderModal({ onKeyCreated });

    fireEvent.change(screen.getByTestId("api-key-name-input"), {
      target: { value: "Windowed" },
    });
    fireEvent.change(screen.getByTestId("api-key-not-before-input"), {
      target: { value: "2026-07-01T10:00" },
    });
    fireEvent.change(screen.getByTestId("api-key-expires-at-input"), {
      target: { value: "2026-08-01T10:00" },
    });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CREATE" }));

    await waitFor(() => {
      expect(mockState.mutateAsync).toHaveBeenCalled();
    });
    const payload = mockState.mutateAsync.mock.calls[0][0];
    expect(payload.name).toBe("Windowed");
    expect(typeof payload.not_before).toBe("string");
    expect(typeof payload.expires_at).toBe("string");
    expect(new Date(payload.not_before).toISOString()).toBe(
      new Date("2026-07-01T10:00").toISOString(),
    );
    expect(new Date(payload.expires_at).toISOString()).toBe(
      new Date("2026-08-01T10:00").toISOString(),
    );
    expect(onKeyCreated).toHaveBeenCalled();
  });

  it("shows an error toast and does not submit when not_before >= expires_at", async () => {
    renderModal({});

    fireEvent.change(screen.getByTestId("api-key-name-input"), {
      target: { value: "Bad Window" },
    });
    fireEvent.change(screen.getByTestId("api-key-not-before-input"), {
      target: { value: "2026-08-01T10:00" },
    });
    fireEvent.change(screen.getByTestId("api-key-expires-at-input"), {
      target: { value: "2026-07-01T10:00" },
    });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CREATE" }));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith(
        "SETTINGS$API_KEY_WINDOW_INVALID",
      );
    });
    expect(mockState.mutateAsync).not.toHaveBeenCalled();
  });

  it("resets all three fields after a successful creation", async () => {
    const onClose = vi.fn();
    renderModal({ onClose });

    fireEvent.change(screen.getByTestId("api-key-name-input"), {
      target: { value: "Test" },
    });
    fireEvent.change(screen.getByTestId("api-key-not-before-input"), {
      target: { value: "2026-07-01T10:00" },
    });
    fireEvent.click(screen.getByRole("button", { name: "BUTTON$CREATE" }));

    await waitFor(() => {
      expect(
        (screen.getByTestId("api-key-name-input") as HTMLInputElement).value,
      ).toBe("");
    });
    expect(
      (screen.getByTestId("api-key-not-before-input") as HTMLInputElement)
        .value,
    ).toBe("");
    expect(
      (screen.getByTestId("api-key-expires-at-input") as HTMLInputElement)
        .value,
    ).toBe("");
    // Modal should NOT auto-close: the parent decides when to switch to
    // the "newly-created key" modal.
    expect(onClose).not.toHaveBeenCalled();
  });
});
