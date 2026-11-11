import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { useSettingsNavItems } from "#/hooks/use-settings-nav-items";
import { WebClientConfig } from "#/api/option-service/option.types";

const useConfigMock = vi.fn();
const useSettingsMock = vi.fn();
const useActiveBackendMock = vi.fn<
  () => { backend: { kind: "local" | "cloud" }; orgId: string | null }
>(() => ({
  backend: { kind: "local" },
  orgId: null,
}));
const useActiveAgentProfileMock = vi.fn<
  () => { activeProfile: { agent_kind: string; name: string } | null }
>(() => ({ activeProfile: null }));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

vi.mock("#/hooks/use-active-agent-profile", () => ({
  useActiveAgentProfile: () => useActiveAgentProfileMock(),
}));

const createConfig = (
  feature_flags: Partial<WebClientConfig["feature_flags"]> = {},
): WebClientConfig => ({
  feature_flags: {
    hide_llm_settings: false,
    hide_users_page: true,
    ...feature_flags,
  },
  providers_configured: [],
  maintenance_start_time: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: new Date().toISOString(),
});

const openHandsSettings = {
  agent_settings: { agent_kind: "openhands" },
};

describe("useSettingsNavItems", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSettingsMock.mockReturnValue({ data: openHandsSettings });
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local" },
      orgId: null,
    });
    useActiveAgentProfileMock.mockReturnValue({ activeProfile: null });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("keeps standalone Agents out of local Settings navigation", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });

    const { result } = renderHook(() => useSettingsNavItems());
    expect(result.current).not.toContainEqual(
      expect.objectContaining({
        item: expect.objectContaining({ to: "/agents" }),
      }),
    );
    expect(result.current).toContainEqual({
      type: "item",
      item: OSS_NAV_ITEMS.find((item) => item.to === "/settings/app"),
    });
    expect(result.current).not.toContainEqual({
      type: "item",
      item: OSS_NAV_ITEMS.find((item) => item.to === "/settings/meta-llm"),
    });
  });

  it("keeps the remaining Settings entries on cloud backends", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud" },
      orgId: "org-123",
    });

    const { result } = renderHook(() => useSettingsNavItems());
    expect(result.current).toEqual(
      OSS_NAV_ITEMS.map((item) => ({ type: "item", item })),
    );
  });

  it("does not list the standalone Agent route on either backend", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });

    for (const kind of ["local", "cloud"] as const) {
      useActiveBackendMock.mockReturnValue({
        backend: { kind },
        orgId: kind === "cloud" ? "org-123" : null,
      });
      const paths = renderHook(() => useSettingsNavItems())
        .result.current.filter((item) => item.type === "item")
        .map((item) => (item.type === "item" ? item.item.to : null));
      expect(paths).not.toContain("/agents");
      expect(paths).not.toContain("/settings/agent");
    }
  });

  it("filters hidden routes from the OSS settings items", () => {
    useConfigMock.mockReturnValue({
      data: createConfig({ hide_llm_settings: true }),
    });

    const { result } = renderHook(() => useSettingsNavItems());
    const paths = result.current
      .filter((item) => item.type === "item")
      .map((item) => (item.type === "item" ? item.item.to : null));

    expect(paths).not.toContain("/settings/llm");
    expect(paths).not.toContain("/settings/secrets");
    expect(paths).not.toContain("/agents");
    expect(paths).toContain("/settings/app");
  });

  it("never lists removed settings sub-pages", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });

    const { result } = renderHook(() => useSettingsNavItems());
    const paths = result.current
      .filter((item) => item.type === "item")
      .map((item) => (item.type === "item" ? item.item.to : null));

    expect(paths).not.toContain("/settings/agent-server");
    expect(paths).not.toContain("/settings/integrations");
    expect(paths).not.toContain("/settings/skills");
    expect(paths).not.toContain("/settings/mcp");
  });

  it("keeps every settings library enabled while an ACP profile is active", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      },
    });
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: { agent_kind: "acp", name: "MyClaude" },
    });

    const { result } = renderHook(() => useSettingsNavItems());
    const byPath = new Map(
      result.current
        .filter((item) => item.type === "item")
        .map(
          (item) => [item.type === "item" ? item.item.to : "", item] as const,
        ),
    );

    for (const path of ["/settings/app", "/settings/verification"]) {
      const renderedItem = byPath.get(path);
      expect(renderedItem).toEqual({
        type: "item",
        item: OSS_NAV_ITEMS.find((item) => item.to === path),
      });
    }
  });

  it("lists the Condenser page only on cloud backends", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });
    const paths = () =>
      renderHook(() => useSettingsNavItems()).result.current.map((item) =>
        item.type === "item" ? item.item.to : null,
      );

    expect(paths()).not.toContain("/settings/condenser");

    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud" },
      orgId: null,
    });
    expect(paths()).toContain("/settings/condenser");
  });

  it("lists only the Application page when the canvas is locked to a Cloud host", () => {
    // Arrange — SaaS / self-hosted OHE serve the canvas with `--lock-to-cloud`;
    // the OHE settings shell ("All Cloud Settings") owns every other page.
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");
    useConfigMock.mockReturnValue({ data: createConfig() });
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud" },
      orgId: "org-123",
    });

    // Act
    const { result } = renderHook(() => useSettingsNavItems());

    // Assert
    const paths = result.current
      .filter((item) => item.type === "item")
      .map((item) => (item.type === "item" ? item.item.to : null));
    expect(paths).toEqual(["/settings/app"]);
  });
});
