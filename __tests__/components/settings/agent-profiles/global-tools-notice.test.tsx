import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GlobalToolsNotice } from "#/components/features/settings/agent-profiles/global-tools-notice";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { names?: string }) =>
      options?.names ? `${key}:${options.names}` : key,
  }),
}));

let agentSettings: Record<string, unknown> | undefined;
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({ data: { agent_settings: agentSettings } }),
}));

const saveSettings = vi.fn();
vi.mock("#/hooks/mutation/use-save-settings", () => ({
  useSaveSettings: () => ({ mutate: saveSettings, isPending: false }),
}));

vi.mock("#/utils/custom-toast-handlers");

describe("GlobalToolsNotice", () => {
  beforeEach(() => {
    saveSettings.mockReset();
  });

  it.each([
    ["unset", { agent_kind: "openhands" }],
    ["null", { agent_kind: "openhands", tools: null }],
    ["an ACP agent", { agent_kind: "acp", tools: [{ name: "terminal" }] }],
  ])("stays hidden when global tools are %s", (_label, settings) => {
    agentSettings = settings;
    render(<GlobalToolsNotice />);
    expect(screen.queryByTestId("global-tools-notice")).toBeNull();
  });

  it("lists a custom global tool list and resets it to the standard set", async () => {
    agentSettings = {
      agent_kind: "openhands",
      tools: [
        { name: "terminal", params: {} },
        { name: "task_tool_set", params: {} },
      ],
    };
    render(<GlobalToolsNotice />);

    expect(screen.getByTestId("global-tools-notice")).toHaveTextContent(
      "terminal, task_tool_set",
    );
    await userEvent.setup().click(screen.getByTestId("global-tools-reset"));

    expect(saveSettings).toHaveBeenCalledWith(
      { agent_settings_diff: { tools: null } },
      expect.any(Object),
    );
  });

  it("shows an explicitly bare global tool list", () => {
    agentSettings = { agent_kind: "openhands", tools: [] };
    render(<GlobalToolsNotice />);
    expect(screen.getByTestId("global-tools-notice")).toBeInTheDocument();
  });
});
