import { describe, expect, it } from "vitest";

import {
  buildProfileToolsValue,
  readProfileTools,
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

  it("keeps the first entry of a repeated tool", () => {
    expect(
      readProfileTools([
        { name: "switch_llm", params: { a: 1 } },
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

  it.each(["task_tool_set", "workflow_tool_set"])(
    "adds task_tracker when %s is selected without it",
    (delegate) => {
      expect(
        buildProfileToolsValue({
          mode: "custom",
          selected: ["terminal", delegate],
        }),
      ).toEqual([
        { name: "terminal", params: {} },
        { name: delegate, params: {} },
        { name: "task_tracker", params: {} },
      ]);
    },
  );

  it("does not duplicate task_tracker or add it without a delegation tool", () => {
    expect(
      buildProfileToolsValue({
        mode: "custom",
        selected: ["task_tool_set", "task_tracker"],
      }),
    ).toHaveLength(2);
    expect(
      buildProfileToolsValue({ mode: "custom", selected: ["terminal"] }),
    ).toEqual([{ name: "terminal", params: {} }]);
  });
});
