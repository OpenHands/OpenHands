import { describe, expect, it, vi } from "vitest";
import { cmdClickAndExpectNewTab } from "../../tests/e2e/mock-llm/utils/cmd-click-new-tab";

describe("cmdClickAndExpectNewTab", () => {
  it("resolves the newly opened tab when waitForEvent succeeds", async () => {
    const mockPage = {
      evaluate: vi.fn().mockResolvedValue(null),
    };
    const mockCard = {
      page: vi.fn().mockReturnValue(mockPage),
      click: vi.fn().mockResolvedValue(undefined),
    };
    const mockNewTab = { url: () => "https://example.com/conversation" };
    const mockContext = {
      waitForEvent: vi.fn().mockResolvedValue(mockNewTab),
    };

    const result = await cmdClickAndExpectNewTab(
      mockContext as any,
      mockCard as any,
      {
        timeout: 5000,
      },
    );

    expect(result).toBe(mockNewTab);
    expect(mockCard.click).toHaveBeenCalledWith({
      modifiers: ["ControlOrMeta"],
    });
    expect(mockContext.waitForEvent).toHaveBeenCalledWith("page", {
      timeout: 5000,
    });
  });

  it("fails fast with recorded click diagnostics when no tab opens within timeout", async () => {
    const diagnostics = {
      targetTag: "A",
      targetTestId: "conversation-card",
      anchorHref: "/conversations/123?backend=b1",
      ctrlKey: true,
      metaKey: false,
      defaultPrevented: false,
    };
    const mockPage = {
      evaluate: vi
        .fn()
        .mockResolvedValueOnce(undefined) // setup click listener
        .mockResolvedValueOnce(diagnostics), // read diagnostics on error
    };
    const mockCard = {
      page: vi.fn().mockReturnValue(mockPage),
      click: vi.fn().mockResolvedValue(undefined),
    };
    const mockContext = {
      waitForEvent: vi
        .fn()
        .mockRejectedValue(new Error("Timeout 5000ms exceeded")),
    };

    await expect(
      cmdClickAndExpectNewTab(mockContext as any, mockCard as any, {
        timeout: 5000,
      }),
    ).rejects.toThrowError(
      /cmd-click opened no new tab within 5000ms\. Recorded click diagnostics: .*conversation-card/,
    );
  });
});
