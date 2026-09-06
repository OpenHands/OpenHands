import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DesktopTitlebar } from "#/components/features/desktop/desktop-titlebar";

describe("DesktopTitlebar", () => {
  it("renders a window-drag region that clears the traffic lights", () => {
    render(<DesktopTitlebar />);

    const bar = screen.getByTestId("desktop-titlebar");
    expect(bar).toHaveAttribute("aria-hidden", "true");
    expect(bar.className).toMatch(/app-region:\s*drag/i);
    expect(bar.className).toMatch(/h-10/);
  });
});
