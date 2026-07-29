import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { GitControlBarPullButton } from "#/components/features/chat/git-control-bar-pull-button";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackPullButtonClick: vi.fn(),
  }),
}));

vi.mock("#/icons/u-arrow-down.svg?react", () => ({
  default: ({ color }: { color?: string }) => (
    <svg data-testid="pull-icon" data-color={color ?? ""} />
  ),
}));

describe("GitControlBarPullButton", () => {
  it("uses muted theme colors when inactive during conversation loading", () => {
    render(
      <GitControlBarPullButton
        onSuggestionsClick={vi.fn()}
        hasRepository
        providerTokensReady
        isConversationReady={false}
      />,
    );

    const button = screen.getByRole("button");
    expect(button).toBeDisabled();
    expect(button.className).toContain("text-[var(--oh-muted)]");
    expect(button.className).toContain("opacity-50");
    expect(button.className).not.toContain("hover:bg-tertiary");
    expect(screen.getByTestId("pull-icon")).toHaveAttribute(
      "data-color",
      "var(--oh-muted)",
    );
  });

  it("uses active theme colors when conversation is ready", () => {
    render(
      <GitControlBarPullButton
        onSuggestionsClick={vi.fn()}
        hasRepository
        providerTokensReady
        isConversationReady
      />,
    );

    const button = screen.getByRole("button");
    expect(button).not.toBeDisabled();
    expect(button.className).toContain("text-white");
    expect(button.className).toContain("hover:bg-tertiary");
    expect(screen.getByTestId("pull-icon")).toHaveAttribute(
      "data-color",
      "white",
    );
  });
});
