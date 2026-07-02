import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreateApiKeyModal } from "./create-api-key-modal";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const mockState = vi.hoisted(() => ({
  mutateAsync: vi.fn(),
  invalidateQueries: vi.fn(),
  isPending: false,
  organizations: {
    organizations: [
      { id: "org-a", name: "Org A", is_personal: false },
      { id: "org-b", name: "Personal Workspace", is_personal: true },
    ],
    currentOrgId: "org-a",
  },
}));

vi.mock("#/hooks/mutation/use-create-api-key", () => ({
  useCreateApiKey: () => ({
    mutateAsync: mockState.mutateAsync,
    isPending: mockState.isPending,
  }),
}));

vi.mock("#/hooks/query/use-organizations", () => ({
  useOrganizations: () => ({
    data: mockState.organizations,
    isLoading: false,
  }),
}));

vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: () => ({ organizationId: "org-a" }),
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
      org_id: null,
    });
  });

  it("renders the new active-window date inputs", () => {
    renderModal({});
    expect(screen.getByTestId("api-key-not-before-input")).toBeInTheDocument();
    expect(screen.getByTestId("api-key-expires-at-input")).toBeInTheDocument();
  });

  it("renders the org selector with the org selector label", () => {
    renderModal({});
    expect(screen.getByTestId("api-key-org-selector")).toBeInTheDocument();
    // The label above the selector makes the "Organization" intent clear.
    expect(screen.getByText("SETTINGS$API_KEY_ORG_LABEL")).toBeInTheDocument();
    // The help text mentions the "All orgs" / "X-Org-Id" trade-off so
    // users discover the new option.
    expect(screen.getByText("SETTINGS$API_KEY_ORG_HELP")).toBeInTheDocument();
  });

  it("submits with name + explicit org_id=null (unbound) when 'All orgs' is selected", async () => {
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
        org_id: null,
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
    expect(payload.org_id).toBeNull();
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
