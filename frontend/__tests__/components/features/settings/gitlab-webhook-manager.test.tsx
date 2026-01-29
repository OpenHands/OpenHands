import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { GitLabWebhookManager } from "#/components/features/settings/git-settings/gitlab-webhook-manager";
import { integrationService } from "#/api/integration-service/integration-service.api";
import type {
  GitLabResource,
  ResourceInstallationResult,
} from "#/api/integration-service/integration-service.types";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

// Mock toast handlers
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

const mockResources: GitLabResource[] = [
  {
    id: "1",
    name: "Test Project",
    full_path: "user/test-project",
    type: "project",
    webhook_installed: false,
    webhook_uuid: null,
    last_synced: null,
  },
  {
    id: "10",
    name: "Test Group",
    full_path: "test-group",
    type: "group",
    webhook_installed: true,
    webhook_uuid: "uuid-123",
    last_synced: "2024-01-01T00:00:00Z",
  },
];

describe("GitLabWebhookManager", () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  beforeEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  const renderComponent = () => {
    return render(
      <QueryClientProvider client={queryClient}>
        <GitLabWebhookManager />
      </QueryClientProvider>,
    );
  };

  it("should display loading state when fetching resources", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockImplementation(
      () => new Promise(() => {}), // Never resolves
    );

    // Act
    renderComponent();

    // Assert
    expect(
      screen.getByText("GITLAB$WEBHOOK_MANAGER_LOADING"),
    ).toBeInTheDocument();
  });

  it("should display error state when fetching fails", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockRejectedValue(
      new Error("Failed to fetch"),
    );

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      expect(
        screen.getByText("GITLAB$WEBHOOK_MANAGER_ERROR"),
      ).toBeInTheDocument();
    });
  });

  it("should display no resources message when list is empty", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [],
    });

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      expect(
        screen.getByText("GITLAB$WEBHOOK_MANAGER_NO_RESOURCES"),
      ).toBeInTheDocument();
    });
  });

  it("should display resources table when resources are available", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: mockResources,
    });

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      expect(screen.getByText("Test Project")).toBeInTheDocument();
      expect(screen.getByText("Test Group")).toBeInTheDocument();
    });

    expect(screen.getByText("user/test-project")).toBeInTheDocument();
    expect(screen.getByText("test-group")).toBeInTheDocument();
  });

  it("should display correct resource types in table", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: mockResources,
    });

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      const projectType = screen.getByText("project");
      const groupType = screen.getByText("group");
      expect(projectType).toBeInTheDocument();
      expect(groupType).toBeInTheDocument();
    });
  });

  it("should disable reinstall button when webhook is already installed", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "10",
          name: "Test Group",
          full_path: "test-group",
          type: "group",
          webhook_installed: true,
          webhook_uuid: "uuid-123",
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      const reinstallButton = screen.getByTestId(
        "reinstall-webhook-button-group:10",
      );
      expect(reinstallButton).toBeDisabled();
    });
  });

  it("should enable reinstall button when webhook is not installed", async () => {
    // Arrange
    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();

    // Assert
    await waitFor(() => {
      const reinstallButton = screen.getByTestId(
        "reinstall-webhook-button-project:1",
      );
      expect(reinstallButton).not.toBeDisabled();
    });
  });

  it("should call reinstall service when reinstall button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const reinstallSpy = vi.spyOn(integrationService, "reinstallGitLabWebhook");

    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();
    const reinstallButton = await screen.findByTestId(
      "reinstall-webhook-button-project:1",
    );
    await user.click(reinstallButton);

    // Assert
    await waitFor(() => {
      expect(reinstallSpy).toHaveBeenCalledWith({
        resource: {
          type: "project",
          id: "1",
        },
      });
    });
  });

  it("should show loading state on button during reinstallation", async () => {
    // Arrange
    const user = userEvent.setup();
    let resolveReinstall: (value: ResourceInstallationResult) => void;
    const reinstallPromise = new Promise<ResourceInstallationResult>(
      (resolve) => {
        resolveReinstall = resolve;
      },
    );

    vi.spyOn(integrationService, "reinstallGitLabWebhook").mockReturnValue(
      reinstallPromise,
    );

    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();
    const reinstallButton = await screen.findByTestId(
      "reinstall-webhook-button-project:1",
    );
    await user.click(reinstallButton);

    // Assert
    await waitFor(() => {
      expect(
        screen.getByText("GITLAB$WEBHOOK_REINSTALLING"),
      ).toBeInTheDocument();
    });

    // Cleanup
    resolveReinstall!({
      resource_id: "1",
      resource_type: "project",
      success: true,
      error: null,
    });
  });

  it("should display error message when reinstallation fails", async () => {
    // Arrange
    const user = userEvent.setup();
    const errorMessage = "Permission denied";
    vi.spyOn(integrationService, "reinstallGitLabWebhook").mockResolvedValue({
      resource_id: "1",
      resource_type: "project",
      success: false,
      error: errorMessage,
    });

    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();
    const reinstallButton = await screen.findByTestId(
      "reinstall-webhook-button-project:1",
    );
    await user.click(reinstallButton);

    // Assert
    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });
  });

  it("should display success toast when reinstallation succeeds", async () => {
    // Arrange
    const user = userEvent.setup();
    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    vi.spyOn(integrationService, "reinstallGitLabWebhook").mockResolvedValue({
      resource_id: "1",
      resource_type: "project",
      success: true,
      error: null,
    });

    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();
    const reinstallButton = await screen.findByTestId(
      "reinstall-webhook-button-project:1",
    );
    await user.click(reinstallButton);

    // Assert
    await waitFor(() => {
      expect(displaySuccessToastSpy).toHaveBeenCalledWith(
        "GITLAB$WEBHOOK_REINSTALL_SUCCESS",
      );
    });
  });

  it("should display error toast when reinstallation throws error", async () => {
    // Arrange
    const user = userEvent.setup();
    const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    const errorMessage = "Network error";

    vi.spyOn(integrationService, "reinstallGitLabWebhook").mockRejectedValue(
      new Error(errorMessage),
    );

    vi.spyOn(integrationService, "getGitLabResources").mockResolvedValue({
      resources: [
        {
          id: "1",
          name: "Test Project",
          full_path: "user/test-project",
          type: "project",
          webhook_installed: false,
          webhook_uuid: null,
          last_synced: null,
        },
      ],
    });

    // Act
    renderComponent();
    const reinstallButton = await screen.findByTestId(
      "reinstall-webhook-button-project:1",
    );
    await user.click(reinstallButton);

    // Assert
    await waitFor(() => {
      expect(displayErrorToastSpy).toHaveBeenCalledWith(errorMessage);
    });
  });
});
