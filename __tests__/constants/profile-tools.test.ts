import { describe, expect, it } from "vitest";

import {
  applyToolSwitchesToProfileTools,
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

  it("reads a built-in's class name as its tool name", () => {
    expect(
      readProfileTools([
        { name: "SwitchLLMTool", params: { a: 1 } },
        { name: "switch_llm" },
      ]),
    ).toEqual({
      mode: "custom",
      selected: ["switch_llm"],
      params: { switch_llm: { a: 1 } },
    });
  });

  it.each(["constructor", "toString", "__proto__"])(
    "keeps a tool named %s",
    (name) => {
      const read = readProfileTools([{ name, params: { x: 1 } }]);
      expect(read.selected).toEqual([name]);
      expect(buildProfileToolsValue({ ...read, mode: "custom" })).toEqual([
        { name, params: { x: 1 } },
      ]);
    },
  );

  it("reads an empty list as a deliberately bare agent, not as standard", () => {
    expect(readProfileTools([])).toMatchObject({
      mode: "custom",
      selected: [],
    });
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

describe("applyToolSwitchesToProfileTools", () => {
  const terminal = { name: "terminal", params: { username: "dev" } };

  it("adds and removes the tools the switches name", () => {
    expect(
      applyToolSwitchesToProfileTools([terminal], {
        subAgents: true,
        switchLlm: true,
      }),
    ).toEqual([
      terminal,
      { name: "task_tool_set", params: {} },
      { name: "switch_llm", params: {} },
    ]);
    expect(
      applyToolSwitchesToProfileTools(
        [
          terminal,
          { name: "task_tool_set", params: {} },
          { name: "SwitchLLMTool", params: {} },
        ],
        { subAgents: false, switchLlm: false },
      ),
    ).toEqual([terminal]);
  });

  it("keeps a stored spelling when the switch agrees with it", () => {
    const stored = [terminal, { name: "SwitchLLMTool", params: {} }];
    expect(
      applyToolSwitchesToProfileTools(stored, {
        subAgents: false,
        switchLlm: true,
      }),
    ).toEqual(stored);
  });

  it("leaves switch_llm alone when that switch is not given", () => {
    expect(
      applyToolSwitchesToProfileTools([terminal], { subAgents: false }),
    ).toEqual([terminal]);
  });
});
