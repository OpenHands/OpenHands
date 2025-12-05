import { describe, it, expect, beforeEach, vi } from "vitest";
import { getObservationContent } from "../get-observation-content";
import { ObservationEvent, OpenHandsEvent, ActionEvent } from "#/types/v1/core";
import { BrowserObservation } from "#/types/v1/core/base/observation";
import { BrowserNavigateAction } from "#/types/v1/core/base/action";
import { SecurityRisk } from "#/types/v1/core/base/common";
import { useBrowserStore } from "#/stores/browser-store";

// Mock the browser store
vi.mock("#/stores/browser-store", () => ({
  useBrowserStore: {
    getState: vi.fn(() => ({
      setScreenshotSrc: vi.fn(),
      setUrl: vi.fn(),
    })),
  },
}));

describe("getObservationContent - BrowserObservation", () => {
  const mockSetScreenshotSrc = vi.fn();
  const mockSetUrl = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockSetScreenshotSrc.mockClear();
    mockSetUrl.mockClear();
    vi.mocked(useBrowserStore.getState).mockReturnValue({
      url: "https://example.com",
      screenshotSrc: "",
      setScreenshotSrc: mockSetScreenshotSrc,
      setUrl: mockSetUrl,
      reset: vi.fn(),
    });
  });

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

    // Should call setScreenshotSrc with properly formatted data URL
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
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
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    );
  });

  it("should not call setScreenshotSrc when screenshot_data is null", () => {
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

    // Should not call setScreenshotSrc when screenshot_data is null
    expect(mockSetScreenshotSrc).not.toHaveBeenCalled();
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
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
      "data:image/png;base64,base64data",
    );
    expect(mockSetUrl).toHaveBeenCalledWith("https://example.com");
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
    expect(mockSetScreenshotSrc).not.toHaveBeenCalled();
    expect(mockSetUrl).not.toHaveBeenCalled();
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
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
      "data:image/png;base64,base64data",
    );
    expect(mockSetUrl).not.toHaveBeenCalled();
    expect(result).toBe("**Output:**\nPage loaded successfully");
  });
});
