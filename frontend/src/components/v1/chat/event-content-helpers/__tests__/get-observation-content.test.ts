import { describe, it, expect } from "vitest";
import { getObservationContent } from "../get-observation-content";
import { ObservationEvent, OpenHandsEvent, ActionEvent } from "#/types/v1/core";
import { BrowserObservation } from "#/types/v1/core/base/observation";
import { BrowserNavigateAction } from "#/types/v1/core/base/action";
import { SecurityRisk } from "#/types/v1/core/base/common";
import { useBrowserStore } from "#/stores/browser-store";

const INITIAL_URL = "https://github.com/OpenHands/OpenHands";

describe("getObservationContent - BrowserObservation", () => {
  it("should update browser store with screenshot data when available", () => {
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
        screenshot_data:
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      },
    };

    const result = getObservationContent(mockEvent);

    // Check actual store state
    expect(useBrowserStore.getState().screenshotSrc).toBe(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    );

    // Should return the output content
    expect(result).toContain("**Output:**");
    expect(result).toContain("Browser action completed");
  });

  it("should handle screenshot data that already has data: prefix", () => {
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
        screenshot_data:
          "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      },
    };

    getObservationContent(mockEvent);

    // Should use the screenshot data as-is since it already has the data: prefix
    expect(useBrowserStore.getState().screenshotSrc).toBe(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    );
  });

  it("should not update screenshotSrc when screenshot_data is null", () => {
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
        screenshot_data: null,
      },
    };

    getObservationContent(mockEvent);

    // screenshotSrc should remain at initial value (empty string)
    expect(useBrowserStore.getState().screenshotSrc).toBe("");
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

    // Should return error content
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

    // Should provide default success message
    expect(result).toBe("Browser action completed successfully.");
  });

  it("should extract URL from most recent browser navigate action when screenshot is available", () => {
    const mockBrowserObservationEvent: ObservationEvent<BrowserObservation> = {
      id: "obs-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "Page loaded successfully",
        error: null,
        screenshot_data: "base64data",
      },
    };

    const mockBrowserNavigateEvent: ActionEvent<BrowserNavigateAction> = {
      id: "action-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "BrowserNavigateAction",
        url: "https://example.com",
        new_tab: false,
      },
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      tool_call: {
        id: "call-id",
        type: "function",
        function: {
          name: "browser_navigate",
          arguments: '{"url": "https://example.com", "new_tab": false}',
        },
      },
      llm_response_id: "response-id",
      security_risk: SecurityRisk.LOW,
    };

    const allEvents: OpenHandsEvent[] = [
      mockBrowserNavigateEvent,
      mockBrowserObservationEvent,
    ];

    const result = getObservationContent(
      mockBrowserObservationEvent,
      allEvents,
    );

    // Should update browser store with screenshot and URL
    const state = useBrowserStore.getState();
    expect(state.screenshotSrc).toBe("data:image/png;base64,base64data");
    expect(state.url).toBe("https://example.com");
    expect(result).toBe("**Output:**\nPage loaded successfully");
  });

  it("should not extract URL when no screenshot data is available", () => {
    const mockBrowserObservationEvent: ObservationEvent<BrowserObservation> = {
      id: "obs-id",
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

    const mockBrowserNavigateEvent: ActionEvent<BrowserNavigateAction> = {
      id: "action-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "agent",
      thought: [],
      thinking_blocks: [],
      action: {
        kind: "BrowserNavigateAction",
        url: "https://example.com",
        new_tab: false,
      },
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      tool_call: {
        id: "call-id",
        type: "function",
        function: {
          name: "browser_navigate",
          arguments: '{"url": "https://example.com", "new_tab": false}',
        },
      },
      llm_response_id: "response-id",
      security_risk: SecurityRisk.LOW,
    };

    const allEvents: OpenHandsEvent[] = [
      mockBrowserNavigateEvent,
      mockBrowserObservationEvent,
    ];

    const result = getObservationContent(
      mockBrowserObservationEvent,
      allEvents,
    );

    // Should not update browser store when no screenshot
    const state = useBrowserStore.getState();
    expect(state.screenshotSrc).toBe("");
    expect(state.url).toBe(INITIAL_URL);
    expect(result).toBe("**Output:**\nPage loaded successfully");
  });

  it("should handle case when no browser navigate action is found", () => {
    const mockBrowserObservationEvent: ObservationEvent<BrowserObservation> = {
      id: "obs-id",
      timestamp: "2024-01-01T00:00:00Z",
      source: "environment",
      tool_name: "browser_navigate",
      tool_call_id: "call-id",
      action_id: "action-id",
      observation: {
        kind: "BrowserObservation",
        output: "Page loaded successfully",
        error: null,
        screenshot_data: "base64data",
      },
    };

    const allEvents: OpenHandsEvent[] = [mockBrowserObservationEvent];

    const result = getObservationContent(
      mockBrowserObservationEvent,
      allEvents,
    );

    // Should update screenshot but not URL
    const state = useBrowserStore.getState();
    expect(state.screenshotSrc).toBe("data:image/png;base64,base64data");
    expect(state.url).toBe(INITIAL_URL);
    expect(result).toBe("**Output:**\nPage loaded successfully");
  });
});
