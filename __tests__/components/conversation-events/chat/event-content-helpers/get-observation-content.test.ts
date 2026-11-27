import { describe, it, expect } from "vitest";
import { getObservationContent } from "#/components/conversation-events/chat/event-content-helpers/get-observation-content";
import { ObservationEvent } from "#/types/agent-server/core";
import {
  BrowserObservation,
  GlobObservation,
  GrepObservation,
} from "#/types/agent-server/core/base/observation";

const makeObservationEvent = (
  observation: Record<string, unknown>,
): ObservationEvent =>
  ({
    id: "observation-formatting",
    timestamp: "2024-01-01T00:00:00Z",
    source: "environment",
    tool_name: "tool",
    tool_call_id: "tool-call",
    action_id: "action",
    observation,
  }) as unknown as ObservationEvent;

const terminalMetadata = (exitCode: number) => ({
  exit_code: exitCode,
  pid: 123,
  username: "openhands",
  hostname: "runtime",
  working_dir: "/workspace",
  py_interpreter_path: null,
  prefix: "",
  suffix: "",
});

describe("getObservationContent - BrowserObservation", () => {
  it("should return output content when available", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "Browser action completed",
        error: null,
        screenshot_data: "base64data",
      },
    };

    const result = getObservationContent(mockEvent);

    expect(result).toContain("**Output:**");
    expect(result).toContain("Browser action completed");
  });

  it("should handle error cases properly", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "",
        error: "Browser action failed",
        screenshot_data: null,
      },
    };

    const result = getObservationContent(mockEvent);

    expect(result).toContain("**Error:**");
    expect(result).toContain("Browser action failed");
  });

  it("should provide default message when no output or error", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "",
        error: null,
        screenshot_data: "base64data",
      },
    };

    const result = getObservationContent(mockEvent);

    expect(result).toBe("Browser action completed successfully.");
  });

  it("should return output when screenshot_data is null", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "Page loaded successfully",
        error: null,
        screenshot_data: null,
      },
    };

    const result = getObservationContent(mockEvent);

    expect(result).toBe("**Output:**\nPage loaded successfully");
  });
});

describe("getObservationContent - GlobObservation", () => {
  it("should display files found when glob matches files", () => {
    // Arrange
    const mockEvent: ObservationEvent<GlobObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "glob",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GlobObservation",
        content: [{ type: "text", text: "Found 2 files", cache_prompt: false }],
        is_error: false,
        files: ["/workspace/src/index.ts", "/workspace/src/app.ts"],
        pattern: "**/*.ts",
        search_path: "/workspace",
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toBe(
      "**Pattern:** `**/*.ts`\n**Search Path:** `/workspace`\n\n**Files Found (2):**\n- `/workspace/src/index.ts`\n- `/workspace/src/app.ts`",
    );
  });

  it("should display no files found message when glob matches nothing", () => {
    // Arrange
    const mockEvent: ObservationEvent<GlobObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "glob",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GlobObservation",
        content: [
          { type: "text", text: "No files found", cache_prompt: false },
        ],
        is_error: false,
        files: [],
        pattern: "**/*.xyz",
        search_path: "/workspace",
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toContain("**Pattern:** `**/*.xyz`");
    expect(result).toContain("**Result:** No files found.");
  });

  it("should display error when glob operation fails", () => {
    // Arrange
    const mockEvent: ObservationEvent<GlobObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "glob",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GlobObservation",
        content: [
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "Permission denied", cache_prompt: false },
          { type: "text", text: "Access blocked", cache_prompt: false },
        ],
        is_error: true,
        files: [],
        pattern: "**/*",
        search_path: "/restricted",
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toBe(
      "**Pattern:** `**/*`\n**Search Path:** `/restricted`\n\n**Error:**\nPermission denied\nAccess blocked",
    );
  });

  it("should indicate truncation when results exceed limit", () => {
    // Arrange
    const mockEvent: ObservationEvent<GlobObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "glob",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GlobObservation",
        content: [{ type: "text", text: "Found files", cache_prompt: false }],
        is_error: false,
        files: ["/workspace/file1.ts"],
        pattern: "**/*.ts",
        search_path: "/workspace",
        truncated: true,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toContain("**Files Found (1+, truncated):**");
  });
});

describe("getObservationContent - GrepObservation", () => {
  it("should display matches found when grep finds results", () => {
    // Arrange
    const mockEvent: ObservationEvent<GrepObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "grep",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GrepObservation",
        content: [
          { type: "text", text: "Found 2 matches", cache_prompt: false },
        ],
        is_error: false,
        matches: ["/workspace/src/api.ts", "/workspace/src/routes.ts"],
        pattern: "fetchData",
        search_path: "/workspace",
        include_pattern: "*.ts",
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toBe(
      "**Pattern:** `fetchData`\n**Search Path:** `/workspace`\n**Include:** `*.ts`\n\n**Matches (2):**\n- `/workspace/src/api.ts`\n- `/workspace/src/routes.ts`",
    );
  });

  it("should display no matches found when grep finds nothing", () => {
    // Arrange
    const mockEvent: ObservationEvent<GrepObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "grep",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GrepObservation",
        content: [{ type: "text", text: "No matches", cache_prompt: false }],
        is_error: false,
        matches: [],
        pattern: "nonExistentFunction",
        search_path: "/workspace",
        include_pattern: null,
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toContain("**Pattern:** `nonExistentFunction`");
    expect(result).toContain("**Result:** No matches found.");
    expect(result).not.toContain("**Include:**");
  });

  it("should display error when grep operation fails", () => {
    // Arrange
    const mockEvent: ObservationEvent<GrepObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "grep",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "GrepObservation",
        content: [
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "Invalid regex pattern", cache_prompt: false },
          { type: "text", text: "Check expression", cache_prompt: false },
        ],
        is_error: true,
        matches: [],
        pattern: "[invalid",
        search_path: "/workspace",
        include_pattern: null,
        truncated: false,
      },
    };

    // Act
    const result = getObservationContent(mockEvent);

    // Assert
    expect(result).toBe(
      "**Pattern:** `[invalid`\n**Search Path:** `/workspace`\n\n**Error:**\nInvalid regex pattern\nCheck expression",
    );
  });
});

describe("getObservationContent - editor and terminal output", () => {
  it("surfaces editor errors without showing stale output", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FileEditorObservation",
          command: "str_replace",
          output: "stale output",
          path: "/workspace/app.ts",
          prev_exist: true,
          old_content: "old",
          new_content: "new",
          error: "The old text was not found",
        }),
      ),
    ).toBe("**Error:**\nThe old text was not found");
  });

  it("renders viewed and successfully edited content in code fences", () => {
    const viewed = getObservationContent(
      makeObservationEvent({
        kind: "FileEditorObservation",
        command: "view",
        output: "fallback",
        content: [
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "1  const value = 1;" },
          { type: "text", text: "2  export default value;" },
        ],
        path: "/workspace/app.ts",
        prev_exist: true,
        old_content: null,
        new_content: null,
        error: null,
      }),
    );
    expect(viewed).toBe(
      "```\n1  const value = 1;\n2  export default value;\n```",
    );

    const edited = getObservationContent(
      makeObservationEvent({
        kind: "StrReplaceEditorObservation",
        command: "str_replace",
        output: "replacement complete",
        content: [{ type: "text", text: "updated file" }],
        path: "/workspace/app.ts",
        prev_exist: true,
        old_content: "before",
        new_content: "after",
        error: null,
      }),
    );
    expect(edited).toBe("```\nupdated file\n```");
  });

  it("falls back to editor output when no text content is available", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FileEditorObservation",
          command: "view",
          output: "directory listing",
          path: "/workspace",
          prev_exist: true,
          old_content: null,
          new_content: null,
          error: null,
        }),
      ),
    ).toBe("```\ndirectory listing\n```");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FileEditorObservation",
          command: "create",
          output: "File created",
          content: [],
          path: "/workspace/new.ts",
          prev_exist: false,
          old_content: null,
          new_content: "",
          error: null,
        }),
      ),
    ).toBe("File created");
  });

  it("handles malformed editor content and one-sided create metadata", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FileEditorObservation",
          command: "view",
          output: "fallback output",
          content: "not-an-array",
          path: "/workspace/app.ts",
          prev_exist: true,
          old_content: null,
          new_content: null,
          error: null,
        }),
      ),
    ).toBe("```\nfallback output\n```");

    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FileEditorObservation",
          command: "create",
          output: "File created",
          content: [{ type: "text", text: "created contents" }],
          path: "/workspace/new.ts",
          prev_exist: false,
          old_content: null,
          new_content: "created contents",
          error: null,
        }),
      ),
    ).toBe("created contents");
  });

  it("shows terminal commands, combines text blocks, and ignores images", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "TerminalObservation",
          content: [
            { type: "text", text: "first" },
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "second" },
          ],
          command: "printf output",
          exit_code: 0,
          is_error: false,
          timeout: false,
          metadata: terminalMetadata(0),
        }),
      ),
    ).toBe("Command: `printf output`\n\nOutput:\n```sh\nfirst\nsecond\n```");
  });

  it("uses the no-output message when a command returns no text", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "ExecuteBashObservation",
        content: [{ type: "image", image_urls: ["ignored"] }],
        command: null,
        exit_code: 0,
        error: false,
        timeout: false,
        metadata: terminalMetadata(0),
      }),
    );

    expect(result).toBe("Output:\n```sh\nOBSERVATION$COMMAND_NO_OUTPUT\n```");

    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "TerminalObservation",
          content: [{ type: "text", text: "   \n  " }],
          command: null,
          exit_code: 0,
          is_error: false,
          timeout: false,
          metadata: terminalMetadata(0),
        }),
      ),
    ).toBe("Output:\n```sh\nOBSERVATION$COMMAND_NO_OUTPUT\n```");
  });

  it("truncates very large terminal output", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "TerminalObservation",
        content: [{ type: "text", text: "x".repeat(1001) }],
        command: "generate-output",
        exit_code: 0,
        is_error: false,
        timeout: false,
        metadata: terminalMetadata(0),
      }),
    );

    expect(result).toContain(`${"x".repeat(1000)}...`);
    expect(result).not.toContain("x".repeat(1001));

    const boundary = getObservationContent(
      makeObservationEvent({
        kind: "TerminalObservation",
        content: [{ type: "text", text: "x".repeat(1000) }],
        command: null,
        exit_code: 0,
        is_error: false,
        timeout: false,
        metadata: terminalMetadata(0),
      }),
    );
    expect(boundary).toBe(`Output:\n\`\`\`sh\n${"x".repeat(1000)}\n\`\`\``);
  });
});

describe("getObservationContent - tool output", () => {
  it("prefers browser text content and ignores non-text blocks", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "BrowserObservation",
        content: [
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "Visible page text" },
          { type: "text", text: "Second line" },
        ],
        output: "legacy output",
        error: null,
        screenshot_data: null,
      }),
    );

    expect(result).toBe("**Output:**\nVisible page text\nSecond line");
  });

  it("falls back to browser output when content has a malformed shape", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "BrowserObservation",
          content: "not-an-array",
          output: "legacy output",
          error: null,
          screenshot_data: null,
        }),
      ),
    ).toBe("**Output:**\nlegacy output");
  });

  it("truncates oversized browser messages", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "BrowserObservation",
        output: "x".repeat(1001),
        error: null,
        screenshot_data: null,
      }),
    );

    expect(result).toHaveLength(1014);
    expect(result.endsWith("...(truncated)")).toBe(true);

    const boundary = getObservationContent(
      makeObservationEvent({
        kind: "BrowserObservation",
        content: [{ type: "text", text: "x".repeat(988) }],
        output: "legacy output",
        error: null,
        screenshot_data: null,
      }),
    );
    expect(boundary).toHaveLength(1000);
    expect(boundary).toBe(`**Output:**\n${"x".repeat(988)}`);
  });

  it("labels successful and failed MCP tool output", () => {
    const successful = getObservationContent(
      makeObservationEvent({
        kind: "MCPToolObservation",
        tool_name: "github_search",
        content: [
          { type: "text", text: "2 issues" },
          { type: "image", image_urls: ["ignored"] },
          { type: "text", text: "1 pull request" },
        ],
        is_error: false,
      }),
    );
    const failed = getObservationContent(
      makeObservationEvent({
        kind: "MCPToolObservation",
        tool_name: "github_search",
        content: [{ type: "text", text: "Access denied" }],
        is_error: true,
      }),
    );

    expect(successful).toBe(
      "**Tool:** github_search\n\n**Result:**\n2 issues\n1 pull request",
    );
    expect(failed).toBe("**Tool:** github_search\n\n**Error:**\nAccess denied");
  });

  it("truncates oversized MCP output", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "MCPToolObservation",
        tool_name: "large_tool",
        content: [{ type: "text", text: "x".repeat(1100) }],
        is_error: false,
      }),
    );

    expect(result).toHaveLength(1003);
    expect(result.endsWith("...")).toBe(true);

    const boundary = getObservationContent(
      makeObservationEvent({
        kind: "MCPToolObservation",
        tool_name: "boundary",
        content: [{ type: "text", text: "x".repeat(968) }],
        is_error: false,
      }),
    );
    expect(boundary).toHaveLength(1000);
    expect(boundary.endsWith("...")).toBe(false);
  });

  it("formats successful and failed skill invocations", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "InvokeSkillObservation",
          skill_name: "testing",
          content: [
            { type: "text", text: "Skill instructions" },
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "Use focused assertions" },
          ],
          is_error: false,
        }),
      ),
    ).toBe(
      "**Skill:** `testing`\n\nSkill instructions\nUse focused assertions",
    );
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "InvokeSkillObservation",
          skill_name: "",
          content: [{ type: "text", text: "Skill unavailable" }],
          is_error: true,
        }),
      ),
    ).toBe("**Error:**\nSkill unavailable");
  });

  it("truncates oversized skill output", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "InvokeSkillObservation",
        skill_name: "testing",
        content: [{ type: "text", text: "x".repeat(1100) }],
        is_error: false,
      }),
    );

    expect(result).toHaveLength(1014);
    expect(result.endsWith("...(truncated)")).toBe(true);

    const boundary = getObservationContent(
      makeObservationEvent({
        kind: "InvokeSkillObservation",
        skill_name: "boundary",
        content: [{ type: "text", text: "x".repeat(977) }],
        is_error: false,
      }),
    );
    expect(boundary).toHaveLength(1000);
    expect(boundary.endsWith("...(truncated)")).toBe(false);
  });

  it("surfaces only Canvas UI acknowledgement text", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "CanvasUIObservation",
          content: [
            { type: "text", text: "Tab opened" },
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "Preview ready" },
          ],
          is_error: false,
        }),
      ),
    ).toBe("Tab opened\nPreview ready");
  });

  it("describes successful model switches with the available context", () => {
    const detailed = getObservationContent(
      makeObservationEvent({
        kind: "SwitchLLMObservation",
        profile_name: "reviewer",
        active_model: "claude-sonnet",
        reason: "Review benefits from a second model",
        content: [],
        is_error: false,
      }),
    );
    expect(detailed).toBe(
      "**Profile:** `reviewer`\n**Active model:** `claude-sonnet`\n**Reason:** Review benefits from a second model",
    );

    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "SwitchLLMObservation",
          profile_name: "fast",
          active_model: null,
          reason: null,
          content: [],
          is_error: false,
        }),
      ),
    ).toBe("**Profile:** `fast`");
  });

  it("uses a server error when present and a useful fallback otherwise", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "SwitchLLMObservation",
          profile_name: "missing",
          active_model: null,
          reason: null,
          content: [
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "Profile not found" },
            { type: "text", text: "Choose another profile" },
          ],
          is_error: true,
        }),
      ),
    ).toBe("**Error:**\nProfile not found\nChoose another profile");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "SwitchLLMObservation",
          profile_name: "missing",
          active_model: null,
          reason: null,
          content: [{ type: "image", image_urls: ["ignored"] }],
          is_error: true,
        }),
      ),
    ).toBe("**Error:**\nFailed to switch LLM profile `missing`.");
  });
});

describe("getObservationContent - task and simple observations", () => {
  it("formats task plans, status icons, notes, and a result message", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "TaskTrackerObservation",
        command: "plan",
        task_list: [
          { title: "Write", notes: "Important", status: "todo" },
          { title: "Test", notes: "", status: "in_progress" },
          { title: "Ship", notes: "", status: "done" },
          { title: "Review", notes: "", status: "blocked" },
        ],
        content: "  Plan updated  ",
      }),
    );

    expect(result).toContain("Task List (4 items)");
    expect(result).toContain("\n1. ⏳ **[TODO]** Write");
    expect(result).toContain("\n2. 🔄 **[IN PROGRESS]** Test");
    expect(result).toContain("⏳ **[TODO]** Write");
    expect(result).toContain("🔄 **[IN PROGRESS]** Test");
    expect(result).toContain("✅ **[DONE]** Ship");
    expect(result).toContain("❓ **[BLOCKED]** Review");
    expect(result).toContain("*Notes: Important*");
    expect(result).not.toContain("*Notes: *");
    expect(result).toContain("**Result:** Plan updated");
  });

  it("uses singular task copy and describes empty plans", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "TaskTrackerObservation",
          command: "plan",
          task_list: [{ title: "Only task", notes: "", status: "todo" }],
          content: "",
        }),
      ),
    ).toContain("Task List (1 item)");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "TaskTrackerObservation",
          command: "plan",
          task_list: [],
          content: "   ",
        }),
      ),
    ).toBe("**Command:** `plan`\n\n**Task List:** Empty");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "TaskTrackerObservation",
          command: "view",
          task_list: [
            { title: "Do not render", notes: "hidden", status: "todo" },
          ],
          content: "Current list",
        }),
      ),
    ).toBe("**Command:** `view`\n\n**Result:** Current list");
  });

  it("returns thought text and ignores image-only thought content", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "ThinkObservation",
          content: [
            { type: "text", text: "Logged" },
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "Still thinking" },
          ],
        }),
      ),
    ).toBe("Logged\nStill thinking");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "ThinkObservation",
          content: [{ type: "image", image_urls: ["ignored"] }],
        }),
      ),
    ).toBe("");
  });

  it("labels failed finish observations and returns successful text directly", () => {
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FinishObservation",
          content: [
            { type: "text", text: "Done" },
            { type: "text", text: "Verified" },
          ],
          is_error: false,
        }),
      ),
    ).toBe("Done\nVerified");
    expect(
      getObservationContent(
        makeObservationEvent({
          kind: "FinishObservation",
          content: [
            { type: "image", image_urls: ["ignored"] },
            { type: "text", text: "Could not finish" },
          ],
          is_error: true,
        }),
      ),
    ).toBe("**Error:**\nCould not finish");
  });

  it("preserves unknown observations as JSON for forward compatibility", () => {
    const content = getObservationContent(
      makeObservationEvent({ kind: "FutureObservation", value: 42 }),
    );

    expect(content).toContain('"kind": "FutureObservation"');
    expect(content).toContain('"value": 42');
  });
});

describe("getObservationContent - large search results", () => {
  it("does not truncate glob output exactly at the content limit", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "GlobObservation",
        content: [],
        is_error: false,
        files: ["x".repeat(936)],
        pattern: "p",
        search_path: "/",
        truncated: false,
      }),
    );

    expect(result).toHaveLength(1000);
    expect(result.endsWith("...(truncated)")).toBe(false);
  });

  it("truncates oversized glob results", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "GlobObservation",
        content: [],
        is_error: false,
        files: [`/workspace/${"x".repeat(1100)}`],
        pattern: "**/*",
        search_path: "/workspace",
        truncated: false,
      }),
    );

    expect(result).toHaveLength(1014);
    expect(result.endsWith("...(truncated)")).toBe(true);
  });

  it("marks truncated grep matches and caps oversized output", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "GrepObservation",
        content: [],
        is_error: false,
        matches: [`/workspace/${"x".repeat(1100)}`],
        pattern: "needle",
        search_path: "/workspace",
        include_pattern: "*.ts",
        truncated: true,
      }),
    );

    expect(result).toContain("**Matches (1+, truncated):**");
    expect(result).toHaveLength(1014);
    expect(result.endsWith("...(truncated)")).toBe(true);
  });

  it("does not truncate grep output exactly at the content limit", () => {
    const result = getObservationContent(
      makeObservationEvent({
        kind: "GrepObservation",
        content: [],
        is_error: false,
        matches: ["x".repeat(940)],
        pattern: "p",
        search_path: "/",
        include_pattern: null,
        truncated: false,
      }),
    );

    expect(result).toHaveLength(1000);
    expect(result.endsWith("...(truncated)")).toBe(false);
  });
});
