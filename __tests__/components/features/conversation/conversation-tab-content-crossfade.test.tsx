import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ConversationTabContentCrossfade } from "#/components/features/conversation/conversation-tabs/conversation-tab-content/conversation-tab-content-crossfade";

describe("ConversationTabContentCrossfade", () => {
  it("renders the tab content when not loading", () => {
    render(
      <ConversationTabContentCrossfade showAgentLoading={false} tabKey="files">
        <div data-testid="tab-child" />
      </ConversationTabContentCrossfade>,
    );

    expect(screen.getByTestId("tab-child")).toBeInTheDocument();
  });
});
