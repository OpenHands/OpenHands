import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiKeysManager } from "./api-keys-manager";
import { ApiKey } from "#/api/api-keys";

const mockApiKeys: ApiKey[] = vi.hoisted(() => [
  {
    id: "1",
    name: "Active Key",
    prefix: "oh_active_",
    created_at: "2026-05-01T10:00:00Z",
    last_used_at: "2026-05-15T10:00:00Z",
    not_before: null,
    expires_at: null,
  },
  {
    id: "2",
    name: "Pending Key",
    prefix: "oh_pending_",
    created_at: "2026-05-01T10:00:00Z",
    last_used_at: null,
    not_before: "2099-01-01T00:00:00Z",
    expires_at: "2099-12-31T00:00:00Z",
  },
  {
    id: "3",
    name: "Expired Key",
    prefix: "oh_expired_",
    created_at: "2025-01-01T10:00:00Z",
    last_used_at: "2025-12-01T10:00:00Z",
    not_before: null,
    expires_at: "2025-12-31T00:00:00Z",
  },
]);

vi.mock("#/hooks/query/use-api-keys", () => ({
  useApiKeys: () => ({ data: mockApiKeys, isLoading: false, error: null }),
}));

vi.mock("#/hooks/query/use-llm-api-key", () => ({
  useLlmApiKey: () => ({
    data: undefined,
    isLoading: true,
    isPaymentRequired: false,
  }),
}));

vi.mock("#/hooks/mutation/use-refresh-llm-api-key", () => ({
  useRefreshLlmApiKey: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

const renderManager = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ApiKeysManager />
    </QueryClientProvider>,
  );

describe("ApiKeysManager - status column", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a Status column header", () => {
    renderManager();
    expect(
      screen.getByRole("columnheader", { name: "SETTINGS$API_KEY_STATUS" }),
    ).toBeInTheDocument();
  });

  it("shows the correct status for an active key", () => {
    renderManager();
    const row = screen.getByText("Active Key").closest("tr");
    expect(row).not.toBeNull();
    expect(row!.querySelector('[class*="bg-green"]')).toBeInTheDocument();
    expect(row!.textContent).toContain("SETTINGS$API_KEY_STATUS_ACTIVE");
  });

  it("shows pending status and dims the row for a future-window key", () => {
    renderManager();
    const row = screen.getByText("Pending Key").closest("tr");
    expect(row).not.toBeNull();
    expect(row!.className).toContain("opacity-60");
    expect(row!.textContent).toContain("SETTINGS$API_KEY_STATUS_PENDING");
    expect(row!.querySelector('[class*="bg-yellow"]')).toBeInTheDocument();
  });

  it("shows expired status and dims the row for a past-expiry key", () => {
    renderManager();
    const row = screen.getByText("Expired Key").closest("tr");
    expect(row).not.toBeNull();
    expect(row!.className).toContain("opacity-60");
    expect(row!.textContent).toContain("SETTINGS$API_KEY_STATUS_EXPIRED");
    expect(row!.querySelector('[class*="bg-red"]')).toBeInTheDocument();
  });

  it("displays the active-window timestamps when set", () => {
    renderManager();
    const row = screen.getByText("Pending Key").closest("tr");
    expect(row).not.toBeNull();
    expect(row!.textContent).toContain("SETTINGS$API_KEY_NOT_BEFORE");
    expect(row!.textContent).toContain("SETTINGS$API_KEY_EXPIRES_AT");
  });
});
