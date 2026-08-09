import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { screen, waitFor } from "@testing-library/react";
import { IntegrationsSettingsScreen } from "#/routes/integrations-settings";
import { useAppwriteIntegration } from "#/hooks/query/use-appwrite-integration";
import { useDependencyTrackIntegration } from "#/hooks/query/use-dependency-track-integration";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useSettings } from "#/hooks/query/use-settings";
import { appwriteApiKeySecretName } from "#/utils/appwrite-integration-secrets";
import { dependencyTrackApiKeySecretName } from "#/utils/dependency-track-integration-secrets";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: {
      kind: "local",
      id: "default-local",
      host: "http://localhost:8000",
    },
  }),
}));

vi.mock("#/hooks/query/use-appwrite-integration", () => ({
  useAppwriteIntegration: vi.fn(),
}));

vi.mock("#/hooks/query/use-dependency-track-integration", () => ({
  useDependencyTrackIntegration: vi.fn(),
}));

vi.mock("#/hooks/query/use-local-workspaces", () => ({
  useLocalWorkspaces: vi.fn(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn(),
}));

vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("#/hooks/mutation/use-create-secret", () => ({
  useCreateSecret: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

const WORKSPACE_ID = "ws-demo";

describe("IntegrationsSettingsScreen", () => {
  beforeEach(() => {
    vi.mocked(useSettings).mockReturnValue({
      data: { integrations: { appwrite: { byWorkspace: {} } } },
      isLoading: false,
    } as unknown as ReturnType<typeof useSettings>);
    vi.mocked(useLocalWorkspaces).mockReturnValue({
      data: {
        workspaces: [
          {
            id: WORKSPACE_ID,
            name: "Demo",
            path: "/workspace/demo",
          },
        ],
        workspaceParents: [],
      },
      isLoading: false,
    } as unknown as ReturnType<typeof useLocalWorkspaces>);
    vi.mocked(useAppwriteIntegration).mockReturnValue({
      workspaceId: WORKSPACE_ID,
      config: {
        enabled: true,
        endpoint: "https://cloud.appwrite.io/v1",
        projectId: "demo",
      },
      apiKeyIsSet: true,
      isReady: true,
      isLoading: false,
      secretName: appwriteApiKeySecretName(WORKSPACE_ID),
    });
    vi.mocked(useDependencyTrackIntegration).mockReturnValue({
      workspaceId: WORKSPACE_ID,
      config: {
        enabled: true,
        baseUrl: "https://dtrack.example.com",
        projectUuid: "proj-uuid-1",
      },
      apiKeyIsSet: true,
      isReady: true,
      isLoading: false,
      secretName: dependencyTrackApiKeySecretName(WORKSPACE_ID),
    });
  });

  it("renders the AppWrite integration form for a workspace", async () => {
    renderWithProviders(<IntegrationsSettingsScreen />);
    await waitFor(() => {
      expect(screen.getByTestId("integrations-settings")).toBeInTheDocument();
    });
    expect(screen.getByTestId("appwrite-integration-card")).toBeInTheDocument();
    expect(screen.getByTestId("appwrite-workspace")).toBeInTheDocument();
    expect(screen.getByTestId("appwrite-enabled")).toBeChecked();
    expect(screen.getByTestId("appwrite-api-key-set")).toBeInTheDocument();
    expect(
      screen.getByTestId("dependency-track-integration-card"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("dependency-track-enabled")).toBeChecked();
    expect(screen.getByTestId("dependency-track-api-key-set")).toBeInTheDocument();
  });

  it("prompts to add a workspace when none exist", async () => {
    vi.mocked(useLocalWorkspaces).mockReturnValue({
      data: { workspaces: [], workspaceParents: [] },
      isLoading: false,
    } as unknown as ReturnType<typeof useLocalWorkspaces>);
    renderWithProviders(<IntegrationsSettingsScreen />);
    await waitFor(() => {
      expect(screen.getByTestId("appwrite-no-workspaces")).toBeInTheDocument();
    });
  });
});
