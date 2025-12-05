import { describe, it, expect, beforeEach, vi } from "vitest";
import { getObservationContent } from "../get-observation-content";
import { ObservationEvent } from "#/types/v1/core";
import { BrowserObservation } from "#/types/v1/core/base/observation";
import { useBrowserStore } from "#/stores/browser-store";

// Mock the browser store
vi.mock("#/stores/browser-store", () => ({
  useBrowserStore: {
    getState: vi.fn(() => ({
      setScreenshotSrc: vi.fn(),
    })),
  },
}));

describe("getObservationContent - BrowserObservation", () => {
  const mockSetScreenshotSrc = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useBrowserStore.getState as any).mockReturnValue({
      setScreenshotSrc: mockSetScreenshotSrc,
    });
  });

  it("should update browser store with screenshot data when available", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      observation: {
        kind: "BrowserObservation",
        output: "Browser action completed",
        error: null,
        screenshot_data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      },
    };

    const result = getObservationContent(mockEvent);

    // Should call setScreenshotSrc with properly formatted data URL
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    );

    // Should return the output content
    expect(result).toContain("**Output:**");
    expect(result).toContain("Browser action completed");
  });

  it("should handle screenshot data that already has data: prefix", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
      observation: {
        kind: "BrowserObservation",
        output: "Browser action completed",
        error: null,
        screenshot_data: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
      },
    };

    getObservationContent(mockEvent);

    // Should use the screenshot data as-is since it already has the data: prefix
    expect(mockSetScreenshotSrc).toHaveBeenCalledWith(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    );
  });

  it("should not call setScreenshotSrc when screenshot_data is null", () => {
    const mockEvent: ObservationEvent<BrowserObservation> = {
      id: "test-id",
      timestamp: "2024-01-01T00:00:00Z",
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
});
