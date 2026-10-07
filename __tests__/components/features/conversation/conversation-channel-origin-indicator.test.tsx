import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  ConversationChannelOriginIndicator,
  getChannelOriginTag,
} from "#/components/features/conversation/conversation-channel-origin-indicator";
import { ConversationTagChips } from "#/components/features/conversation-panel/conversation-card/conversation-tag-chips";

describe("getChannelOriginTag", () => {
  it("returns the origin tag when present", () => {
    expect(getChannelOriginTag({ origin: "slack", owner: "alice" })).toEqual({
      key: "origin",
      value: "slack",
    });
  });

  it("falls back to the source tag when no origin is stamped", () => {
    expect(getChannelOriginTag({ source: "discord" })).toEqual({
      key: "source",
      value: "discord",
    });
  });

  it("prefers origin over source when both are stamped", () => {
    expect(getChannelOriginTag({ origin: "slack", source: "discord" })).toEqual(
      { key: "origin", value: "slack" },
    );
  });

  it("normalizes the key so cloud-stamped casing still resolves", () => {
    expect(getChannelOriginTag({ Origin: "slack" })).toEqual({
      key: "Origin",
      value: "slack",
    });
  });

  it("returns null when no channel tag is present", () => {
    expect(getChannelOriginTag({ owner: "alice", env: "prod" })).toBeNull();
    expect(getChannelOriginTag(null)).toBeNull();
    expect(getChannelOriginTag(undefined)).toBeNull();
    expect(getChannelOriginTag({})).toBeNull();
  });

  it("ignores reserved keys", () => {
    expect(
      getChannelOriginTag({ acpserver: "claude-code", git_provider: "github" }),
    ).toBeNull();
  });
});

describe("ConversationChannelOriginIndicator", () => {
  it("renders the label and value-specific icon for an origin tag", () => {
    renderWithProviders(
      <ConversationChannelOriginIndicator tags={{ origin: "slack" }} />,
    );

    const indicator = screen.getByTestId("conversation-channel-origin");
    expect(indicator).toHaveTextContent("Origin: slack");
    expect(indicator).toHaveAttribute("title", "Origin: slack");
    // The accessible name must carry the channel value, not just the key, or a
    // screen reader would read only "Origin".
    expect(indicator).toHaveAttribute("aria-label", "Origin: slack");
    expect(indicator).toHaveAttribute("data-tag-key", "origin");

    const icon = screen.getByTestId("conversation-channel-origin-icon");
    expect(icon).toHaveAttribute("data-tag-key", "origin");
    // SlackIcon renders an SVG mark, proving the value-specific icon resolved.
    expect(icon.tagName.toLowerCase()).toBe("svg");
  });

  it("matches the conversation-list chip's label and icon for the same tag", () => {
    renderWithProviders(
      <div>
        <ConversationTagChips tags={[["origin", "slack"]]} />
        <ConversationChannelOriginIndicator tags={{ origin: "slack" }} />
      </div>,
    );

    const listChip = screen.getByTestId("conversation-card-tag-chip");
    const listIcon = within(listChip).getByTestId(
      "conversation-card-tag-chip-icon",
    );
    const indicator = screen.getByTestId("conversation-channel-origin");
    const indicatorIcon = screen.getByTestId(
      "conversation-channel-origin-icon",
    );

    expect(indicator.textContent).toBe(listChip.textContent);
    expect(indicatorIcon.tagName).toBe(listIcon.tagName);
    expect(indicatorIcon.getAttribute("class")).toBe(
      listIcon.getAttribute("class"),
    );
    // Same mark (path data), not just the same wrapper element.
    expect(indicatorIcon.innerHTML).toBe(listIcon.innerHTML);
    expect(indicatorIcon.innerHTML.length).toBeGreaterThan(0);
  });

  it("renders no indicator without a channel tag", () => {
    const { container } = renderWithProviders(
      <ConversationChannelOriginIndicator tags={{ owner: "alice" }} />,
    );

    expect(container).toBeEmptyDOMElement();
    expect(
      screen.queryByTestId("conversation-channel-origin"),
    ).not.toBeInTheDocument();
  });
});
