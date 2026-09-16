import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PinToggleIcon } from "#/components/shared/pin-toggle-icon";

describe("PinToggleIcon", () => {
  it("renders the filled pin at the same size as the outline pin", () => {
    const { container: pinned } = render(<PinToggleIcon pinned />);
    const { container: unpinned } = render(<PinToggleIcon pinned={false} />);

    expect(pinned.querySelector("svg")).toHaveClass("h-4.5", "w-4.5");
    expect(unpinned.querySelector("svg")).toHaveClass("h-4.5", "w-4.5");
  });
});
