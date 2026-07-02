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
    org_id: null,
  },
  {
    id: "2",
    name: "Pending Key",
    prefix: "oh_pending_",
    created_at: "2026-05-01T10:00:00Z",
    last_used_at: null,
    not_before: "2099-01-01T00:00:00Z",
    expires_at: "2099-12-31T00:00:00Z",
    org_id: null,
  },
  {
    id: "3",
    name: "Acme Key",
    prefix: "oh_acme_",
    created_at: "2025-01-01T10:00:00Z",
    last_used_at: "2025-12-01T10:00:00Z",
    not_before: null,
    expires_at: null,
    org_id: "org-acme",
  },
  {
    id: "4",
    name: "Personal Key",
    prefix: "oh_personal_",
    created_at: "2025-01-01T10:00:00Z",
    last_used_at: null,
    not_before: null,
    expires_at: null,
    org_id: "org-personal",
  },
  {
    id: "5",
    name: "Orphaned Key",
    prefix: "oh_orphan_",
    created_at: "2025-01-01T10:00:00Z",
    last_used_at: null,
    not_before: null,
    expires_at: null,
    org_id: "org-no-longer-a-member",
  },
  {
    // The expired-status test below expects a key whose name is
    // "Expired Key" and whose expires_at is in the past; restore
    // that row so the status tests still find it.
    id: "6",
    name: "Expired Key",
    prefix: "oh_expired_",
    created_at: "2025-01-01T10:00:00Z",
    last_used_at: "2025-12-01T10:00:00Z",
    not_before: null,
    expires_at: "2025-12-31T00:00:00Z",
    org_id: "org-acme",
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

vi.mock("#/hooks/query/use-organizations", () => ({
  useOrganizations: () => ({
    data: {
      organizations: [
        // Order matters: the test for "Personal Key" expects the
        // localized "Personal Workspace" label, matching the
        // ``is_personal: true`` flag.
        { id: "org-acme", name: "Acme Inc.", is_personal: false },
        { id: "org-personal", name: "ignored", is_personal: true },
      ],
      currentOrgId: "org-acme",
    },
    isLoading: false,
  }),
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

describe("ApiKeysManager - scope column", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a Scope column header", () => {
    renderManager();
    expect(
      screen.getByRole("columnheader", { name: "SETTINGS$API_KEY_SCOPE" }),
    ).toBeInTheDocument();
  });

  it("labels unbound keys (org_id=null) with the 'All orgs' badge", () => {
    renderManager();
    const row = screen.getByText("Active Key").closest("tr");
    expect(row).not.toBeNull();
    expect(row!.textContent).toContain("SETTINGS$API_KEY_SCOPE_ALL_ORGS");
    // Unbound badges use the blue accent color.
    expect(row!.querySelector('[class*="bg-blue"]')).toBeInTheDocument();
  });

  it("labels bound keys with the org's display name", () => {
    renderManager();
    const row = screen.getByText("Acme Key").closest("tr");
    expect(row).not.toBeNull();
    // The bound badge shows the org name, not the generic "Bound to
    // current org" string. The tooltip carries the descriptive label.
    expect(row!.textContent).toContain("Acme Inc.");
    expect(row!.textContent).not.toContain("SETTINGS$API_KEY_SCOPE_BOUND");
    // The bound-badge span (gray "tertiary" background) carries the
    // explanatory tooltip; assert on it specifically rather than the
    // generic [title] selector which would also match the name cell.
    const badge = row!.querySelector('span[title^="SETTINGS$API_KEY_SCOPE"]');
    expect(badge?.getAttribute("title")).toBe(
      "SETTINGS$API_KEY_SCOPE_BOUND_TITLE",
    );
    // Bound badges do not use the blue accent color.
    expect(row!.querySelector('[class*="bg-blue"]')).toBeNull();
  });

  it("renders 'Personal Workspace' for keys bound to the personal org", () => {
    renderManager();
    const row = screen.getByText("Personal Key").closest("tr");
    expect(row).not.toBeNull();
    // The badge uses the localized "Personal Workspace" label rather
    // than the underlying org name, matching the create modal and the
    // header org selector.
    expect(row!.textContent).toContain("ORG$PERSONAL_WORKSPACE");
    expect(row!.textContent).not.toContain("ignored");
  });

  it("falls back to a short id when the bound org is no longer visible", () => {
    renderManager();
    const row = screen.getByText("Orphaned Key").closest("tr");
    expect(row).not.toBeNull();
    // The user is no longer a member of this org so we have no
    // display name; show a short id prefix and the descriptive title.
    expect(row!.textContent).toContain("org-no-l");
  });
});
