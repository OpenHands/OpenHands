import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
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

  it("returns the LLM settings item unchanged on local backends", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });

    const { result } = renderHook(() => useSettingsNavItems());
    const llmItem = result.current.find(
      (item) => item.type === "item" && item.item.to === "/settings/llm",
    );

    const baseLlm = OSS_NAV_ITEMS.find((item) => item.to === "/settings/llm")!;
    expect(llmItem).toEqual({
      type: "item",
      item: baseLlm,
    });
  });

  it("keeps the generic LLM settings item on cloud backends", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "cloud" },
      orgId: "org-123",
    });

    const { result } = renderHook(() => useSettingsNavItems());

    // Agent profiles are available on cloud too (OpenHands #15060), so every
    // OSS item is present; only the `/settings` LLM-Profiles rename stays
    // local-only, so on cloud every item is passed through unchanged.
    expect(result.current).toEqual(
      OSS_NAV_ITEMS.map((item) => ({ type: "item", item })),
    );
  });

  it("shows a single Agent item (the profile library) on both local and cloud", () => {
    useConfigMock.mockReturnValue({ data: createConfig() });

    for (const kind of ["local", "cloud"] as const) {
      useActiveBackendMock.mockReturnValue({
        backend: { kind },
        orgId: kind === "cloud" ? "org-123" : null,
      });
      const paths = renderHook(() => useSettingsNavItems())
        .result.current.filter((item) => item.type === "item")
        .map((item) => (item.type === "item" ? item.item.to : null));
      // "Agent" IS the profile library; the old standalone /settings/agent
      // global-agent form + separate /settings/agents "Agent profiles" item
      // were collapsed into one.
      expect(paths).toContain("/settings/agents");
      expect(paths).not.toContain("/settings/agent");
      expect(paths.filter((p) => p === "/settings/agents")).toHaveLength(1);
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
    expect(paths).toContain("/settings/app");
    expect(paths).toContain("/settings/secrets");
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

    for (const path of [
      "/settings/llm",
      "/settings/condenser",
      "/settings/verification",
    ]) {
      const renderedItem = byPath.get(path);
      expect(renderedItem).toEqual({
        type: "item",
        item: OSS_NAV_ITEMS.find((item) => item.to === path),
      });
    }
  });
});
