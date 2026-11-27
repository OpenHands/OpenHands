import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import { McpLogoStackBadge } from "#/components/features/mcp-page/mcp-logo-stack-badge";

function entry(id: string) {
  const match = INTEGRATION_CATALOG.find((item) => item.id === id);
  if (!match) {
    throw new Error(`Missing MCP catalog entry: ${id}`);
  }
  return match;
}

describe("McpLogoStackBadge", () => {
  it("renders a full-size badge for a single MCP", () => {
    render(
      <McpLogoStackBadge
        testId="automation-icon"
        entries={[entry("github")]}
      />,
    );

    const icon = screen.getByTestId("automation-icon");
    expect(icon).toHaveClass("h-10", "w-10");
    expect(icon).not.toHaveAttribute("data-layout");
  });

  it("overlaps two MCP logos on a shared grey square", () => {
    render(
      <McpLogoStackBadge
        testId="automation-icon"
        entries={[entry("tavily"), entry("notion")]}
      />,
    );

    const icon = screen.getByTestId("automation-icon");
    expect(icon).toHaveAttribute("data-layout", "overlap");
    expect(icon).toHaveClass("bg-surface-raised");
  });

  it("lays out three or four MCP logos in quadrants", () => {
    render(
      <McpLogoStackBadge
        testId="automation-icon"
        entries={[entry("slack"), entry("linear"), entry("notion")]}
      />,
    );

    expect(screen.getByTestId("automation-icon")).toHaveAttribute(
      "data-layout",
      "quadrants",
    );
  });
});
