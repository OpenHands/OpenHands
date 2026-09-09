import { describe, expect, it } from "vitest";
import {
  buildProfileToolCatalog,
  buildProfileToolsValue,
  readProfileTools,
  standardProfileToolNames,
} from "#/constants/profile-tools";

const KNOWN = [
  "terminal",
  "file_editor",
  "task_tracker",
  "glob",
  "grep",
  "browser_tool_set",
];

describe("buildProfileToolCatalog", () => {
  it("offers every described tool when the backend advertises nothing", () => {
    expect(buildProfileToolCatalog({ usableTools: null })).toEqual(KNOWN);
  });

  it("drops described tools the backend cannot run", () => {
    const catalog = buildProfileToolCatalog({
      usableTools: ["terminal", "file_editor", "task_tracker"],
    });
    expect(catalog).toEqual(["terminal", "file_editor", "task_tracker"]);
  });

  it("appends advertised tools it has no description for, after the known ones", () => {
    const catalog = buildProfileToolCatalog({
      usableTools: ["workflow", "terminal", "ask_oracle"],
    });
    expect(catalog).toEqual(["terminal", "ask_oracle", "workflow"]);
  });

  it("keeps stored names the backend does not advertise, so a save can't drop them", () => {
    const catalog = buildProfileToolCatalog({
      usableTools: ["terminal"],
      storedToolNames: ["terminal", "some_removed_tool"],
    });
    expect(catalog).toEqual(["terminal", "some_removed_tool"]);
  });

  it("never offers the sub-agent tool — the enable_sub_agents toggle owns it", () => {
    const catalog = buildProfileToolCatalog({
      usableTools: ["terminal", "task", "task_tool_set"],
      storedToolNames: ["task_tool_set"],
    });
    expect(catalog).toEqual(["terminal"]);
  });

  it("never offers tools another mechanism already attaches", () => {
    // The advertised list a live agent-server actually returns.
    const catalog = buildProfileToolCatalog({
      usableTools: [
        "canvas_ui",
        "canvas_ui_control",
        "FinishTool",
        "ThinkTool",
        "SwitchLLMTool",
        "InvokeSkillTool",
        "VisionInspectTool",
        "terminal",
      ],
    });
    expect(catalog).toEqual(["terminal"]);
  });
});

describe("standardProfileToolNames", () => {
  it("mirrors the SDK default plus the server's browser injection", () => {
    expect(
      standardProfileToolNames({ usableTools: null, subAgentsEnabled: false }),
    ).toEqual(["terminal", "file_editor", "task_tracker", "browser_tool_set"]);
  });

  it("omits browser when the backend says it is not usable", () => {
    expect(
      standardProfileToolNames({
        usableTools: ["terminal", "file_editor", "task_tracker"],
        subAgentsEnabled: false,
      }),
    ).toEqual(["terminal", "file_editor", "task_tracker"]);
  });

  it("adds the sub-agent tool when delegation is on", () => {
    expect(
      standardProfileToolNames({ usableTools: null, subAgentsEnabled: true }),
    ).toContain("task_tool_set");
  });
});

describe("readProfileTools", () => {
  it("reads a null/absent field as the standard set", () => {
    expect(readProfileTools(null)).toEqual({
      mode: "standard",
      selected: [],
      params: {},
    });
    expect(readProfileTools(undefined).mode).toBe("standard");
  });

  it("reads an empty list as an explicitly bare agent, not as standard", () => {
    expect(readProfileTools([])).toEqual({
      mode: "custom",
      selected: [],
      params: {},
    });
  });

  it("hides the sub-agent tool from the selection but keeps its params", () => {
    const read = readProfileTools([
      { name: "terminal", params: {} },
      { name: "task_tool_set", params: { limit: 2 } },
    ]);
    expect(read.selected).toEqual(["terminal"]);
    expect(read.params.task_tool_set).toEqual({ limit: 2 });
  });

  it("ignores entries without a string name and de-duplicates", () => {
    const read = readProfileTools([
      { name: "terminal", params: { a: 1 } },
      { name: "terminal", params: { a: 2 } },
      { params: {} },
      "terminal",
    ]);
    expect(read.selected).toEqual(["terminal"]);
    expect(read.params.terminal).toEqual({ a: 1 });
  });
});

describe("buildProfileToolsValue", () => {
  const base = {
    selected: ["terminal", "grep"],
    subAgentsEnabled: false,
    usableTools: null,
  };

  it("persists null for the standard set", () => {
    expect(buildProfileToolsValue({ ...base, mode: "standard" })).toBeNull();
  });

  it("persists the selection with its stored params", () => {
    expect(
      buildProfileToolsValue({
        ...base,
        mode: "custom",
        params: { grep: { max_results: 50 } },
      }),
    ).toEqual([
      { name: "terminal", params: {} },
      { name: "grep", params: { max_results: 50 } },
    ]);
  });

  it("re-adds the sub-agent tool so delegation survives an explicit list", () => {
    const tools = buildProfileToolsValue({
      ...base,
      mode: "custom",
      subAgentsEnabled: true,
    });
    expect(tools?.map((tool) => tool.name)).toEqual([
      "terminal",
      "grep",
      "task_tool_set",
    ]);
  });

  it("does not add the sub-agent tool the backend cannot run", () => {
    const tools = buildProfileToolsValue({
      ...base,
      mode: "custom",
      subAgentsEnabled: true,
      usableTools: ["terminal", "grep"],
    });
    expect(tools?.map((tool) => tool.name)).toEqual(["terminal", "grep"]);
  });

  it("persists an empty list for a deliberately bare agent", () => {
    expect(
      buildProfileToolsValue({ ...base, mode: "custom", selected: [] }),
    ).toEqual([]);
  });
});
