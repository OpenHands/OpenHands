import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BitbucketDCWebhookManager } from "#/components/features/settings/git-settings/bitbucket-dc-webhook-manager";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type { BitbucketDCResource } from "#/api/integration-service/integration-service.types";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const mockResources: BitbucketDCResource[] = [
  {
    project_key: "PROJ",
    repo_slug: "myrepo",
    name: "myrepo",
    full_name: "PROJ/myrepo",
    type: "repository",
    connection_id: null,
    webhook_enrolled: false,
    webhook_id: null,
    webhook_url: null,
    webhook_secret_set: false,
    installed_by_user_id: null,
    last_synced: null,
  },
  {
    project_key: "OPS",
    repo_slug: "platform",
    name: "platform",
    full_name: "OPS/platform",
    type: "repository",
    connection_id: 7,
    webhook_enrolled: true,
    webhook_id: "42",
    webhook_url:
      "https://example.com/integration/bitbucket-dc/connections/7/events",
    webhook_secret_set: true,
    installed_by_user_id: "kc-bot",
    last_synced: "2026-01-01T00:00:00",
  },
];

describe("BitbucketDCWebhookManager", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    vi.clearAllMocks();
  });

  const renderComponent = () =>
    render(
      <QueryClientProvider client={queryClient}>
        <BitbucketDCWebhookManager />
      </QueryClientProvider>,
    );

  it("renders repositories with enrollment status", async () => {
    vi.spyOn(integrationService, "getBitbucketDCResources").mockResolvedValue({
      resources: mockResources,
    });

    renderComponent();

    await waitFor(() => {
      expect(screen.getByText("myrepo")).toBeInTheDocument();
      expect(screen.getByText("platform")).toBeInTheDocument();
    });

    expect(screen.getByText("PROJ/myrepo")).toBeInTheDocument();
    expect(screen.getByText("OPS/platform")).toBeInTheDocument();
    expect(
      screen.getByText(
        I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_STATUS_NOT_ENROLLED,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_STATUS_ENROLLED),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.BITBUCKET_DATA_CENTER$WEBHOOK_ENROLLED_BY),
    ).toBeInTheDocument();
  });

  it("calls reinstall when the Install button is clicked on a not-enrolled repo", async () => {
    const user = userEvent.setup();
    vi.spyOn(integrationService, "getBitbucketDCResources").mockResolvedValue({
      resources: mockResources,
    });
    const reinstallSpy = vi
      .spyOn(integrationService, "reinstallBitbucketDCWebhook")
      .mockResolvedValue({
        project_key: "PROJ",
        repo_slug: "myrepo",
        success: true,
        error: null,
        webhook_id: "101",
        connection_id: 8,
        webhook_url:
          "https://example.com/integration/bitbucket-dc/connections/8/events",
      });

    renderComponent();

    await user.click(
      await screen.findByTestId("bbdc-install-webhook-PROJ/myrepo"),
    );

    await waitFor(() => {
      expect(reinstallSpy).toHaveBeenCalledWith({
        resource: { project_key: "PROJ", repo_slug: "myrepo" },
      });
    });
  });

  it("calls uninstall when the Uninstall button is clicked on an enrolled repo", async () => {
    const user = userEvent.setup();
    vi.spyOn(integrationService, "getBitbucketDCResources").mockResolvedValue({
      resources: mockResources,
    });
    const uninstallSpy = vi
      .spyOn(integrationService, "uninstallBitbucketDCWebhook")
      .mockResolvedValue({
        project_key: "OPS",
        repo_slug: "platform",
        success: true,
        error: null,
        webhook_id: "42",
        connection_id: 7,
        webhook_url:
          "https://example.com/integration/bitbucket-dc/connections/7/events",
      });

    renderComponent();

    await user.click(
      await screen.findByTestId("bbdc-uninstall-webhook-OPS/platform"),
    );

    await waitFor(() => {
      expect(uninstallSpy).toHaveBeenCalledWith({
        resource: { project_key: "OPS", repo_slug: "platform" },
      });
    });
  });

  it("does not render an Uninstall button on a not-enrolled repo", async () => {
    vi.spyOn(integrationService, "getBitbucketDCResources").mockResolvedValue({
      resources: mockResources,
    });

    renderComponent();

    await screen.findByTestId("bbdc-install-webhook-PROJ/myrepo");
    expect(
      screen.queryByTestId("bbdc-uninstall-webhook-PROJ/myrepo"),
    ).not.toBeInTheDocument();
  });
});
