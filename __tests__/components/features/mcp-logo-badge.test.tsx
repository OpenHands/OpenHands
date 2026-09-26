import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  McpLogoBadge,
  type McpLogoEntry,
} from "#/components/features/mcp-logo-badge";

const entry = (overrides: Partial<McpLogoEntry>): McpLogoEntry => ({
  id: "github",
  name: "GitHub",
  iconBg: "var(--oh-surface)",
  logoUrl: "https://cdn.simpleicons.org/github/FFFFFF",
  ...overrides,
});

describe("McpLogoBadge", () => {
  it("paints a white mark on a theme surface with the theme ink", () => {
    render(<McpLogoBadge entry={entry({})} testId="badge" />);

    const mark = screen.getByTestId("mcp-logo-tinted-mark");
    expect(mark.style.maskImage).toBe(
      'url("https://cdn.simpleicons.org/github/FFFFFF")',
    );
    expect(mark).toHaveClass("bg-current");
    expect(screen.getByTestId("badge").style.color).toBe("var(--oh-contrast)");
    expect(screen.queryByRole("img")).toBeNull();
  });

  it("keeps white marks as images on brand backgrounds", () => {
    render(
      <McpLogoBadge
        entry={entry({
          id: "linear",
          name: "Linear",
          iconBg: "#5E6AD2",
          logoUrl: "https://cdn.simpleicons.org/linear/FFFFFF",
        })}
        testId="badge"
      />,
    );

    expect(screen.queryByTestId("mcp-logo-tinted-mark")).toBeNull();
    expect(screen.getByAltText("Linear logo")).toHaveAttribute(
      "src",
      "https://cdn.simpleicons.org/linear/FFFFFF",
    );
    expect(screen.getByTestId("badge").style.color).toBe("rgb(255, 255, 255)");
  });

  it("keeps brand-colored marks on theme surfaces as images", () => {
    render(
      <McpLogoBadge
        entry={entry({
          id: "gitlab",
          name: "GitLab",
          logoUrl: "https://cdn.simpleicons.org/gitlab/FC6D26",
        })}
      />,
    );

    expect(screen.queryByTestId("mcp-logo-tinted-mark")).toBeNull();
    expect(screen.getByAltText("GitLab logo")).toBeInTheDocument();
  });
});
