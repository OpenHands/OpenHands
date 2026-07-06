import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JiraDcIntegrationPanel } from "./jira-dc-integration-panel";

// Mock the I18n hooks and translation keys
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
  initReactI18next: {
    type: "3rdParty",
    init: vi.fn(),
  },
}));

// We can define variables that tests can override
let mockConfig = {
  jira_dc_oauth_host: null as string | null,
  jira_dc_host: null as string | null,
  jira_dc_service_account_managed: false,
  jira_dc_service_account_email: "",
};

let mockIntegrationData = {
  status: "inactive",
  workspace: null as Record<string, unknown> | null,
};

let mockHasPermission = true;

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: mockConfig }),
}));

vi.mock("#/hooks/query/use-integration-status", () => ({
  useIntegrationStatus: () => ({ data: mockIntegrationData }),
}));

vi.mock("#/hooks/mutation/use-configure-integration", () => ({
  useConfigureIntegration: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/mutation/use-link-integration", () => ({
  useLinkIntegration: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/mutation/use-unlink-integration", () => ({
  useUnlinkIntegration: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/mutation/use-update-jira-dc-workspace-status", () => ({
  useUpdateJiraDcWorkspaceStatus: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/mutation/use-validate-integration", () => ({
  useValidateIntegration: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("#/hooks/query/use-me", () => ({
  useMe: () => ({ data: { role: "admin" } }),
}));

vi.mock("#/hooks/organizations/use-permissions", () => ({
  usePermission: () => ({
    hasPermission: () => mockHasPermission,
  }),
}));

vi.mock("#/hooks/query/use-jira-dc-instance-status", () => ({
  useJiraDcInstanceStatus: () => ({
    data: { configured: false, host: null },
  }),
}));

const renderPanel = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <JiraDcIntegrationPanel />
    </QueryClientProvider>,
  );

describe("JiraDcIntegrationPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockConfig = {
      jira_dc_oauth_host: null,
      jira_dc_host: null,
      jira_dc_service_account_managed: false,
      jira_dc_service_account_email: "",
    };
    mockIntegrationData = {
      status: "inactive",
      workspace: null,
    };
    mockHasPermission = true;
  });

  it("Managed + email-match: Connect becomes enabled", async () => {
    // 1. Managed + email-match (jira_dc_oauth_host: null, jira_dc_service_account_managed: true)
    mockConfig.jira_dc_oauth_host = null;
    mockConfig.jira_dc_host = "jira.example.com";
    mockConfig.jira_dc_service_account_managed = true;
    mockConfig.jira_dc_service_account_email = "service@example.com";

    renderPanel();

    // Click Configure
    const configureBtn = screen.getByTestId("jira-dc-configure-button");
    fireEvent.click(configureBtn);

    // Dialog opens. Admin PAT for auto-install (which is default)
    // Find Connect button
    const submitBtn = screen.getByTestId("jira-dc-submit-button");

    // Initially disabled because auto-install requires admin PAT
    expect(submitBtn).toBeDisabled();

    // Fill in admin PAT
    const patInput = screen.getByTestId("admin-api-key-input");
    fireEvent.change(patInput, { target: { value: "my-admin-pat" } });

    // Now it should be enabled!
    expect(submitBtn).not.toBeDisabled();

    // Switch to manual mode
    const manualBtn = screen.getByTestId("webhook-mode-manual");
    fireEvent.click(manualBtn);

    // In manual mode, it should be enabled directly because it just generates details
    expect(submitBtn).not.toBeDisabled();
  });

  it("Unmanaged + email-match: host input is rendered; Connect stays disabled until host filled", async () => {
    // 2. Unmanaged + email-match
    mockConfig.jira_dc_oauth_host = null;
    mockConfig.jira_dc_host = null;
    mockConfig.jira_dc_service_account_managed = false;

    renderPanel();

    const configureBtn = screen.getByTestId("jira-dc-configure-button");
    fireEvent.click(configureBtn);

    // Host input is rendered
    const hostInput = screen.getByTestId("jira-dc-host-input");
    expect(hostInput).toBeInTheDocument();

    const submitBtn = screen.getByTestId("jira-dc-submit-button");
    expect(submitBtn).toBeDisabled();

    // Fill service account
    const emailInput = screen.getByTestId("jira-dc-svc-email-input");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    const patInput = screen.getByTestId("jira-dc-svc-pat-input");
    fireEvent.change(patInput, { target: { value: "svc-pat" } });

    // Fill admin PAT
    const adminPatInput = screen.getByTestId("admin-api-key-input");
    fireEvent.change(adminPatInput, { target: { value: "admin-pat" } });

    // Still disabled because host is empty
    expect(submitBtn).toBeDisabled();

    // Fill host
    fireEvent.change(hostInput, { target: { value: "jira.example.com" } });

    // Now it should be enabled
    expect(submitBtn).not.toBeDisabled();
  });

  it("OAuth mode: existing behavior preserved", async () => {
    // 3. OAuth mode
    mockConfig.jira_dc_oauth_host = "oauth.example.com";
    mockConfig.jira_dc_host = "oauth.example.com";
    mockConfig.jira_dc_service_account_managed = false;

    renderPanel();

    // Entry action is Connect (direct OAuth link)
    const connectBtn = screen.getByTestId("jira-dc-connect-button");
    expect(connectBtn).toBeInTheDocument();
    expect(connectBtn).not.toBeDisabled();
  });
});
