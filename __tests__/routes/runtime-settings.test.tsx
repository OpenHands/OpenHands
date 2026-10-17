import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import RuntimeSettingsScreen from "#/routes/runtime-settings";
import type { SettingsValue } from "#/types/settings";

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
});
