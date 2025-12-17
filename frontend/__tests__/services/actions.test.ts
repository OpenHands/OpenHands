import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { handleStatusMessage } from "#/services/actions";
import { StatusMessage } from "#/types/message";
import { queryClient } from "#/query-client-config";
import { useStatusStore } from "#/state/status-store";
import { trackError } from "#/utils/error-handler";

// Mock dependencies
vi.mock("#/query-client-config", () => ({
  queryClient: {
    invalidateQueries: vi.fn(),
  },
}));

vi.mock("#/state/status-store", () => ({
  useStatusStore: {
    getState: vi.fn(() => ({
      setCurStatusMessage: vi.fn(),
    })),
  },
}));

vi.mock("#/utils/error-handler", () => ({
  trackError: vi.fn(),
}));

describe("handleStatusMessage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should invalidate queries when receiving a conversation title update", () => {
    // Create a status message with a conversation title
    const statusMessage: StatusMessage = {
      status_update: true,
      type: "info",
      message: "conversation-123",
      conversation_title: "New Conversation Title",
    };

    // Call the function
    handleStatusMessage(statusMessage);

    // Verify that queryClient.invalidateQueries was called with the correct parameters
    expect(queryClient.invalidateQueries).toHaveBeenCalledWith({
      queryKey: ["user", "conversation", "conversation-123"],
    });
  });

  it("should call setCurStatusMessage for info messages without conversation_title", () => {
    // Create a status message without a conversation title
    const statusMessage: StatusMessage = {
      status_update: true,
      type: "info",
      message: "Some info message",
    };

    const mockSetCurStatusMessage = vi.fn();
    vi.mocked(useStatusStore.getState).mockReturnValue({
      setCurStatusMessage: mockSetCurStatusMessage,
      curStatusMessage: {
        status_update: true,
        type: "info",
        id: "",
        message: "",
      },
    });

    // Call the function
    handleStatusMessage(statusMessage);

    // Verify that setCurStatusMessage was called with the correct message
    expect(mockSetCurStatusMessage).toHaveBeenCalledWith(statusMessage);

    // Verify that queryClient.invalidateQueries was not called
    expect(queryClient.invalidateQueries).not.toHaveBeenCalled();
  });

  it("should call trackError for error messages", () => {
    // Create an error status message
    const statusMessage: StatusMessage = {
      status_update: true,
      type: "error",
      id: "ERROR_ID",
      message: "Some error message",
    };

    // Call the function
    handleStatusMessage(statusMessage);

    // Verify that trackError was called with the correct parameters
    expect(trackError).toHaveBeenCalledWith({
      message: "Some error message",
      source: "chat",
      metadata: { msgId: "ERROR_ID" },
    });

    // Verify that queryClient.invalidateQueries was not called
    expect(queryClient.invalidateQueries).not.toHaveBeenCalled();
  });
});
