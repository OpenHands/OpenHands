import { describe, expect, it } from "vitest";
import {
  parseLoadedResources,
  toLoadedHookResources,
  toLoadedSkillResources,
} from "#/utils/loaded-resources";

describe("parseLoadedResources", () => {
  it("extracts the conversation's hook commands and enabled MCPs", () => {
    const resources = parseLoadedResources({
      agent: {
        agent_context: {
          skills: [
            {
              name: "zebra",
              description: null,
              source: "project/zebra.md",
            },
            {
              name: "alpha",
              description: "Alpha guidance",
              source: "public/alpha/SKILL.md",
            },
          ],
        },
        mcp_config: {
          github: { command: "npx" },
          docs: { url: "https://example.com/mcp", transport: "sse" },
          disabled: { command: "disabled-mcp", enabled: false },
        },
      },
      hook_config: {
        pre_tool_use: [
          {
            matcher: "*",
            hooks: [{ command: "lint" }, { command: "test" }],
          },
        ],
        stop: [{ matcher: "*", hooks: [{ command: "notify" }] }],
      },
    });

    expect(resources).toEqual({
      skills: [
        {
          name: "alpha",
          description: "Alpha guidance",
          source: "public/alpha/SKILL.md",
        },
        {
          name: "zebra",
          description: null,
          source: "project/zebra.md",
        },
      ],
      hooks: [
        { hookType: "pre_tool_use", commands: ["lint", "test"] },
        { hookType: "stop", commands: ["notify"] },
      ],
      mcps: [
        { name: "github", transport: "stdio" },
        { name: "docs", transport: "sse" },
      ],
    });
  });

  it("maps and sorts Cloud-loaded skill records", () => {
    expect(
      toLoadedSkillResources([
        {
          name: "zebra",
          type: "knowledge",
          source: "project/zebra.md",
          content: "---\ndescription: Zebra guidance\n---\nBody",
        },
        {
          name: "alpha",
          type: "agentskills",
          source: null,
          description: "Alpha guidance",
        },
      ]),
    ).toEqual([
      {
        name: "alpha",
        description: "Alpha guidance",
        source: null,
      },
      {
        name: "zebra",
        description: "Zebra guidance",
        source: "project/zebra.md",
      },
    ]);
  });

  it("flattens shared hook responses and counts individual commands", () => {
    expect(
      toLoadedHookResources({
        hooks: [
          {
            event_type: "pre_tool_use",
            matchers: [
              {
                matcher: "terminal",
                hooks: [
                  { type: "command", command: "lint", timeout: 60 },
                  { type: "command", command: "test", timeout: 60 },
                ],
              },
              { matcher: "browser", hooks: undefined },
            ],
          },
          {
            event_type: "stop",
            matchers: [{ matcher: "*", hooks: [] }],
          },
        ],
      }),
    ).toEqual([{ hookType: "pre_tool_use", commands: ["lint", "test"] }]);
  });

  it("keeps unsupported serialized categories unavailable", () => {
    expect(
      parseLoadedResources({
        agent: { agent_context: { skills: [] } },
      }),
    ).toEqual({
      skills: [],
      hooks: null,
      mcps: null,
    });
  });

  it("preserves explicitly empty serialized categories", () => {
    expect(
      parseLoadedResources({
        agent: { agent_context: { skills: [] }, mcp_config: {} },
        hook_config: {},
      }),
    ).toEqual({
      skills: [],
      hooks: [],
      mcps: [],
    });
  });

  it("rejects a missing serialized skill snapshot", () => {
    expect(() => parseLoadedResources({})).toThrow(
      "Loaded resource data is unavailable",
    );
    expect(() => parseLoadedResources({ agent: {} })).toThrow(
      "Loaded skill data is unavailable",
    );
  });
});
