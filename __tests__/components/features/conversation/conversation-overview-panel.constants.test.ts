import { describe, expect, it } from "vitest";
import {
  CONVERSATION_OVERVIEW_COLUMN_MIN_GAP_PX,
  CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX,
  CONVERSATION_OVERVIEW_MIN_THREAD_WIDTH_PX,
  CONVERSATION_OVERVIEW_THREAD_MAX_WIDTH_PX,
  getConversationOverviewLayoutMode,
  hasEnoughOverviewLayoutSpace,
  hasOverviewOverlayLayoutSpace,
} from "#/components/features/conversation/conversation-overview-panel.constants";

describe("conversation overview layout mode", () => {
  const inlineMinWidth =
    CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX +
    CONVERSATION_OVERVIEW_COLUMN_MIN_GAP_PX +
    CONVERSATION_OVERVIEW_MIN_THREAD_WIDTH_PX;

  const overlayMinWidth =
    CONVERSATION_OVERVIEW_THREAD_MAX_WIDTH_PX +
    2 *
      (CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX +
        CONVERSATION_OVERVIEW_COLUMN_MIN_GAP_PX);

  it("hides overview when the container is too narrow for any layout", () => {
    expect(hasEnoughOverviewLayoutSpace(inlineMinWidth - 1)).toBe(false);
    expect(getConversationOverviewLayoutMode(inlineMinWidth - 1)).toBe(
      "hidden",
    );
  });

  it("uses inline layout when overview fits only by pushing the thread", () => {
    expect(getConversationOverviewLayoutMode(inlineMinWidth)).toBe("inline");
    expect(getConversationOverviewLayoutMode(overlayMinWidth - 1)).toBe(
      "inline",
    );
  });

  it("uses overlay layout when overview fits in the centered thread margin", () => {
    expect(hasOverviewOverlayLayoutSpace(overlayMinWidth)).toBe(true);
    expect(getConversationOverviewLayoutMode(overlayMinWidth)).toBe("overlay");
    expect(getConversationOverviewLayoutMode(overlayMinWidth + 200)).toBe(
      "overlay",
    );
  });
});
