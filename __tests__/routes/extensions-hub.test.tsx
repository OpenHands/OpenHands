import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockNavigate = vi.fn();
const mockUseBreakpoint = vi.fn();

vi.mock("react-router", () => ({
  Navigate: (props: { to: string; replace?: boolean }) => {
    mockNavigate(props);
    return null;
  },
}));

vi.mock("#/hooks/use-breakpoint", () => ({
  useBreakpoint: () => mockUseBreakpoint(),
}));

vi.mock("#/components/features/skills/extensions-mobile-hub", () => ({
  ExtensionsMobileHub: () => <div data-testid="extensions-mobile-hub" />,
}));

import ExtensionsHub from "#/routes/extensions-hub";

describe("ExtensionsHub", () => {
  beforeEach(() => {
    mockNavigate.mockReset();
    mockUseBreakpoint.mockReturnValue(false);
  });

  it("redirects desktop Customize navigation to MCP Servers", () => {
    render(<ExtensionsHub />);

    expect(mockNavigate).toHaveBeenCalledWith({ to: "/mcp", replace: true });
  });
});
