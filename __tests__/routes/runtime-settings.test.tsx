import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import RuntimeSettingsScreen from "#/routes/runtime-settings";
import type { SettingsValue } from "#/types/settings";

const mockUseConversationWorkspace = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/query/use-conversation-workspace", () => ({
  useConversationWorkspace: () => mockUseConversationWorkspace(),
}));

const RUNTIME_OPTION_IDS = [
  "sdk-settings-worktree_base",
  "sdk-settings-worktree_base_ref",
  "sdk-settings-worktree_branch_prefix",
  "sdk-settings-container_memory",
  "sdk-settings-container_cpus",
  "sdk-settings-container_browser",
];

async function renderRuntimeOptions(
  conversationSettings: Record<string, SettingsValue>,
) {
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...MOCK_DEFAULT_USER_SETTINGS,
    conversation_settings: conversationSettings,
  });
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<RuntimeSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  await screen.findByTestId("sdk-settings-default_runtime");
  return RUNTIME_OPTION_IDS.filter((id) => screen.queryByTestId(id));
}

describe("RuntimeSettingsScreen", () => {
  beforeEach(() => {
    mockUseConversationWorkspace.mockReturnValue({
      dockerSelectable: true,
      serverInfoLoading: false,
    });
  });

  it.each<[string, Record<string, SettingsValue>, string[]]>([
    ["local", { default_runtime: "local" }, []],
    [
      "worktree",
      { default_runtime: "worktree" },
      ["sdk-settings-worktree_base", "sdk-settings-worktree_branch_prefix"],
    ],
    [
      "worktree from a ref",
      { default_runtime: "worktree", worktree_base: "ref" },
      [
        "sdk-settings-worktree_base",
        "sdk-settings-worktree_base_ref",
        "sdk-settings-worktree_branch_prefix",
      ],
    ],
    [
      "docker",
      { default_runtime: "docker" },
      [
        "sdk-settings-container_memory",
        "sdk-settings-container_cpus",
        "sdk-settings-container_browser",
      ],
    ],
  ])("shows only the %s runtime's options", async (_, settings, expected) => {
    expect(await renderRuntimeOptions(settings)).toEqual(expected);
  });

  describe("on a server that cannot run Docker", () => {
    beforeEach(() => {
      mockUseConversationWorkspace.mockReturnValue({
        dockerSelectable: false,
        serverInfoLoading: false,
      });
    });

    it("does not offer Docker as the default runtime", async () => {
      await renderRuntimeOptions({ default_runtime: "local" });
      await userEvent.click(screen.getByTestId("sdk-settings-default_runtime"));
      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual([
        "SCHEMA$DEFAULT_RUNTIME$CHOICE$LOCAL",
        "SCHEMA$DEFAULT_RUNTIME$CHOICE$WORKTREE",
      ]);
      expect(
        screen.queryByTestId("runtime-docker-unavailable"),
      ).not.toBeInTheDocument();
    });

    it("hides the container options and explains a saved Docker default", async () => {
      expect(await renderRuntimeOptions({ default_runtime: "docker" })).toEqual(
        [],
      );
      expect(
        screen.getByTestId("runtime-docker-unavailable"),
      ).toBeInTheDocument();
    });
  });
});
