import { describe, expect, it, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useAppwriteIntegration } from "#/hooks/query/use-appwrite-integration";
import { useSettings } from "#/hooks/query/use-settings";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { appwriteApiKeySecretName } from "#/utils/appwrite-integration-secrets";

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: vi.fn(),
}));

vi.mock("#/hooks/query/use-get-secrets", () => ({
  useSearchSecrets: vi.fn(),
}));

const WORKSPACE_ID = "ws-demo";

describe("useAppwriteIntegration", () => {
  it("isReady only when workspace config, project, and API key secret exist", () => {
    vi.mocked(useSettings).mockReturnValue({
      data: {
        integrations: {
          appwrite: {
            byWorkspace: {
              [WORKSPACE_ID]: {
                enabled: true,
                endpoint: "https://cloud.appwrite.io/v1",
                projectId: "proj",
              },
            },
          },
        },
      },
      isLoading: false,
    } as unknown as ReturnType<typeof useSettings>);
    vi.mocked(useSearchSecrets).mockReturnValue({
      data: [{ name: appwriteApiKeySecretName(WORKSPACE_ID) }],
      isLoading: false,
    } as unknown as ReturnType<typeof useSearchSecrets>);

    const { result } = renderHook(() => useAppwriteIntegration(WORKSPACE_ID));
    expect(result.current.isReady).toBe(true);
    expect(result.current.apiKeyIsSet).toBe(true);
    expect(result.current.workspaceId).toBe(WORKSPACE_ID);
  });

  it("is not ready without API key secret", () => {
    vi.mocked(useSettings).mockReturnValue({
      data: {
        integrations: {
          appwrite: {
            byWorkspace: {
              [WORKSPACE_ID]: {
                enabled: true,
                endpoint: "https://cloud.appwrite.io/v1",
                projectId: "proj",
              },
            },
          },
        },
      },
      isLoading: false,
    } as unknown as ReturnType<typeof useSettings>);
    vi.mocked(useSearchSecrets).mockReturnValue({
      data: [],
      isLoading: false,
    } as unknown as ReturnType<typeof useSearchSecrets>);

    const { result } = renderHook(() => useAppwriteIntegration(WORKSPACE_ID));
    expect(result.current.isReady).toBe(false);
  });

  it("is not ready without a workspace id", () => {
    vi.mocked(useSettings).mockReturnValue({
      data: {
        integrations: {
          appwrite: {
            byWorkspace: {
              [WORKSPACE_ID]: {
                enabled: true,
                endpoint: "https://cloud.appwrite.io/v1",
                projectId: "proj",
              },
            },
          },
        },
      },
      isLoading: false,
    } as unknown as ReturnType<typeof useSettings>);
    vi.mocked(useSearchSecrets).mockReturnValue({
      data: [{ name: appwriteApiKeySecretName(WORKSPACE_ID) }],
      isLoading: false,
    } as unknown as ReturnType<typeof useSearchSecrets>);

    const { result } = renderHook(() => useAppwriteIntegration(null));
    expect(result.current.isReady).toBe(false);
  });
});
