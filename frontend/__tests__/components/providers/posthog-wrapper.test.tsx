import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { PostHogWrapper } from "#/components/providers/posthog-wrapper";
import OptionService from "#/api/option-service/option-service.api";

// Mock PostHogProvider to capture the options passed to it
const mockPostHogProvider = vi.fn();
vi.mock("posthog-js/react", () => ({
  PostHogProvider: (props: Record<string, unknown>) => {
    mockPostHogProvider(props);
    return props.children;
  },
}));

describe("PostHogWrapper", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset URL hash
    window.location.hash = "";
    // Mock the config fetch
    // @ts-expect-error - partial mock
    vi.spyOn(OptionService, "getConfig").mockResolvedValue({
      POSTHOG_CLIENT_KEY: "test-posthog-key",
    });
  });

  it("should initialize PostHog with bootstrap IDs from URL hash", async () => {
    // Set up URL hash with cross-domain tracking params
    window.location.hash = "ph_distinct_id=user-123&ph_session_id=session-456";

    render(
      <PostHogWrapper>
        <div data-testid="child" />
      </PostHogWrapper>,
    );

    // Wait for async config fetch and PostHog initialization
    await screen.findByTestId("child");

    // Verify PostHogProvider was called with bootstrap options
    expect(mockPostHogProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        options: expect.objectContaining({
          bootstrap: {
            distinctID: "user-123",
            sessionID: "session-456",
          },
        }),
      }),
    );
  });

  it("should clean up URL hash after extracting bootstrap IDs", async () => {
    // Set up URL hash with cross-domain tracking params
    window.location.hash = "ph_distinct_id=user-123&ph_session_id=session-456";

    render(
      <PostHogWrapper>
        <div data-testid="child" />
      </PostHogWrapper>,
    );

    // Wait for async config fetch and PostHog initialization
    await screen.findByTestId("child");

    // Verify URL hash was cleaned up
    expect(window.location.hash).toBe("");
  });
});
