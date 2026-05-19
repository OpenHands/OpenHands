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
    webhook_enrolled: false,
    webhook_id: null,
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
    webhook_enrolled: true,
    webhook_id: "42",
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

  it("shows manual setup values after enrolling a repo", async () => {
    const user = userEvent.setup();
    vi.spyOn(integrationService, "getBitbucketDCResources").mockResolvedValue({
      resources: mockResources,
    });
    vi.spyOn(integrationService, "enrollBitbucketDCWebhook").mockResolvedValue({
      project_key: "PROJ",
      repo_slug: "myrepo",
      success: true,
      error: null,
      webhook_url: "https://ohe.example.com/integration/bitbucket-dc/events",
      webhook_secret: "generated-secret",
      webhook_name: "OpenHands Resolver",
      events: ["pr:comment:added", "pr:comment:edited"],
    });

    renderComponent();

    await user.click(
      await screen.findByTestId("bbdc-enroll-webhook-PROJ/myrepo"),
    );

    await waitFor(() => {
      expect(screen.getByText("generated-secret")).toBeInTheDocument();
    });

    expect(screen.getByText("OpenHands Resolver")).toBeInTheDocument();
    expect(
      screen.getByText(
        "https://ohe.example.com/integration/bitbucket-dc/events",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("pr:comment:added, pr:comment:edited"),
    ).toBeInTheDocument();
  });
});
