import { describe, it, expect } from "vitest";
import { getObservationContent } from "#/components/v1/chat/event-content-helpers/get-observation-content";
import { ObservationEvent } from "#/types/v1/core";
import {
  BrowserObservation,
  GlobObservation,
  GrepObservation,
} from "#/types/v1/core/base/observation";

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
    expect(result).toContain("**Pattern:** `**/*.ts`");
    expect(result).toContain("**Search Path:** `/workspace`");
    expect(result).toContain("**Files Found (2):**");
    expect(result).toContain("- `/workspace/src/index.ts`");
    expect(result).toContain("- `/workspace/src/app.ts`");
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
        content: [{ type: "text", text: "No files found", cache_prompt: false }],
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
        content: [{ type: "text", text: "Permission denied", cache_prompt: false }],
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
    expect(result).toContain("**Error:**");
    expect(result).toContain("Permission denied");
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
        content: [{ type: "text", text: "Found 2 matches", cache_prompt: false }],
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
    expect(result).toContain("**Pattern:** `fetchData`");
    expect(result).toContain("**Search Path:** `/workspace`");
    expect(result).toContain("**Include:** `*.ts`");
    expect(result).toContain("**Matches (2):**");
    expect(result).toContain("- `/workspace/src/api.ts`");
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
        content: [{ type: "text", text: "Invalid regex pattern", cache_prompt: false }],
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
    expect(result).toContain("**Error:**");
    expect(result).toContain("Invalid regex pattern");
  });
});
