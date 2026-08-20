import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TextShimmer } from "#/components/shared/text-shimmer";

vi.mock("framer-motion", async () => {
  const actual = await vi.importActual<typeof import("framer-motion")>(
    "framer-motion",
  );
  return {
    ...actual,
    useReducedMotion: () => false,
  };
});

describe("TextShimmer", () => {
  it("renders with a clipped dual-layer shimmer background", () => {
    render(
      <TextShimmer data-testid="shimmer" duration={3} spread={2}>
        Sending...
      </TextShimmer>,
    );

    const shimmer = screen.getByTestId("shimmer");
    expect(shimmer.style.backgroundImage).toContain("repeating-linear-gradient");
    expect(shimmer.style.backgroundImage).toContain("var(--oh-foreground)");
    expect(shimmer.style.backgroundImage).toContain("var(--oh-muted)");
    expect(shimmer.style.backgroundSize).toBe("200% 100%");
    expect(shimmer.style.animation).toContain("oh-text-shimmer-");
  });
});
