import { describe, it, expect } from "vitest";
import { getObservationContent } from "#/components/v1/chat/event-content-helpers/get-observation-content";
import { ObservationEvent } from "#/types/v1/core";
import { BrowserObservation } from "#/types/v1/core/base/observation";

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
