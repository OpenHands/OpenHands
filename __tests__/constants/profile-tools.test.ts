import { describe, expect, it } from "vitest";

import {
  buildProfileToolsValue,
  readProfileTools,
  toolSwitchesFromProfileTools,
} from "#/constants/profile-tools";

describe("readProfileTools", () => {
  it.each([[null], [undefined], ["terminal"], [{}]])(
    "reads %s as the server's standard set",
    (value) => {
      expect(readProfileTools(value)).toEqual({
        mode: "standard",
        selected: [],
        params: {},
      });
    },
  );

  it("reads an explicit list, keeping params the editor does not model", () => {
    expect(
      readProfileTools([
        { name: "terminal", params: { username: "dev" } },
        { name: "glob" },
      ]),
    ).toEqual({
      mode: "custom",
      selected: ["terminal", "glob"],
      params: { terminal: { username: "dev" }, glob: {} },
    });
  });

  it("reads an empty list as a deliberately bare agent, not as standard", () => {
    expect(readProfileTools([])).toMatchObject({ mode: "custom", selected: [] });
  });
});

describe("buildProfileToolsValue", () => {
  it("saves null for standard so the server keeps deciding", () => {
    expect(
      buildProfileToolsValue({ mode: "standard", selected: ["terminal"] }),
    ).toBeNull();
  });

  it("saves the selection with its stored params", () => {
    expect(
      buildProfileToolsValue({
        mode: "custom",
        selected: ["terminal", "glob"],
        params: { terminal: { username: "dev" } },
      }),
    ).toEqual([
      { name: "terminal", params: { username: "dev" } },
      { name: "glob", params: {} },
    ]);
  });

  it("round-trips an explicit selection", () => {
    const stored = [{ name: "glob", params: {} }];
    const { mode, selected, params } = readProfileTools(stored);
    expect(buildProfileToolsValue({ mode, selected, params })).toEqual(stored);
  });
});

describe("toolSwitchesFromProfileTools", () => {
  it("reads unset tools as the legacy defaults", () => {
    expect(toolSwitchesFromProfileTools(null)).toEqual({
      enable_sub_agents: false,
      enable_switch_llm_tool: true,
    });
  });

  it.each([
    [["terminal"], false, false],
    [["terminal", "task_tool_set", "switch_llm"], true, true],
    [["SwitchLLMTool"], false, true],
  ])("reads %j as the switches it implies", (names, subAgents, switchLlm) => {
    expect(
      toolSwitchesFromProfileTools(names.map((name) => ({ name }))),
    ).toEqual({
      enable_sub_agents: subAgents,
      enable_switch_llm_tool: switchLlm,
    });
  });
});
