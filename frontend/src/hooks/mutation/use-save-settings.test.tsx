import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePostHog } from "posthog-js/react";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSettings } from "../query/use-settings";
import { useSaveSettings } from "./use-save-settings";

vi.mock("posthog-js/react", () => ({
  usePostHog: vi.fn(),
}));
vi.mock("#/context/use-selected-organization", () => ({
  useSelectedOrganizationId: vi.fn(),
}));
vi.mock("../query/use-settings", () => ({
  useSettings: vi.fn(),
}));

const mockedUsePostHog = vi.mocked(usePostHog);
const mockedUseSelectedOrganizationId = vi.mocked(useSelectedOrganizationId);
const mockedUseSettings = vi.mocked(useSettings);

const createWrapper = (queryClient: QueryClient) =>
  function TestWrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

describe("useSaveSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    mockedUsePostHog.mockReturnValue({
      capture: vi.fn(),
    } as never);
    mockedUseSelectedOrganizationId.mockReturnValue({
      organizationId: "org-123",
      setOrganizationId: vi.fn(),
    });
    mockedUseSettings.mockReturnValue({
      data: null,
    } as never);
  });

  it("filters org saves down to explicit diff and org update keys", async () => {
    const queryClient = new QueryClient();
    const saveOrgSpy = vi
      .spyOn(organizationService, "saveOrganizationAgentSettings")
      .mockResolvedValue({
        agent_settings: {},
        conversation_settings: {},
        llm_api_key_set: false,
      });

    const { result } = renderHook(() => useSaveSettings("org"), {
      wrapper: createWrapper(queryClient),
    });

    await act(async () => {
      await result.current.mutateAsync({
        agent_settings_diff: {
          llm: {
            model: "claude-opus-4-5-20251101",
            api_key: "  test-key  ",
          },
        },
        conversation_settings_diff: {
          confirmation_mode: true,
        },
        search_api_key: "  search-key  ",
        llm_api_key: undefined,
        agent_settings_schema: undefined,
        conversation_settings_schema: undefined,
        agent_settings: { ignored: true },
        conversation_settings: { ignored: true },
        git_user_name: "ignored",
      });
    });

    expect(saveOrgSpy).toHaveBeenCalledWith({
      agent_settings_diff: {
        llm: {
          model: "claude-opus-4-5-20251101",
          api_key: "test-key",
        },
      },
      conversation_settings_diff: {
        confirmation_mode: true,
      },
      search_api_key: "search-key",
    });
  });

  it("invalidates both org and personal settings queries after an org save", async () => {
    const queryClient = new QueryClient();
    const invalidateQueriesSpy = vi.spyOn(queryClient, "invalidateQueries");
    vi.spyOn(
      organizationService,
      "saveOrganizationAgentSettings",
    ).mockResolvedValue({
      agent_settings: {},
      conversation_settings: {},
      llm_api_key_set: false,
    });

    const { result } = renderHook(() => useSaveSettings("org"), {
      wrapper: createWrapper(queryClient),
    });

    await act(async () => {
      await result.current.mutateAsync({
        agent_settings_diff: { llm: { model: "claude-opus-4-5-20251101" } },
      });
    });

    await waitFor(() => {
      expect(invalidateQueriesSpy).toHaveBeenCalledTimes(2);
    });

    expect(invalidateQueriesSpy).toHaveBeenNthCalledWith(1, {
      queryKey: ["settings", "org", "org-123"],
    });
    expect(invalidateQueriesSpy).toHaveBeenNthCalledWith(2, {
      queryKey: ["settings", "personal", "org-123"],
    });
  });
});
