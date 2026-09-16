import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";

describe("KeyStatusIcon", () => {
  it("exposes accessible text for the set state by default", () => {
    render(<KeyStatusIcon isSet />);
    const icon = screen.getByTestId("set-indicator");
    expect(icon).toHaveAttribute("aria-label", "API key set");
    expect(icon).toHaveAttribute("title", "API key set");
  });

  it("exposes accessible text for the unset state by default", () => {
    render(<KeyStatusIcon isSet={false} />);
    const icon = screen.getByTestId("unset-indicator");
    expect(icon).toHaveAttribute("aria-label", "API key not set");
    expect(icon).toHaveAttribute("title", "API key not set");
  });

  it("uses a caller-supplied label instead of the generic default", () => {
    // A standalone status icon (no adjacent key field for context, e.g. a
    // provider connection row) needs a label that doesn't imply more than
    // "a key string is stored" — this never pings the provider.
    const label = "API key stored (not verified against the provider)";
    render(<KeyStatusIcon isSet label={label} />);
    const icon = screen.getByTestId("set-indicator");
    expect(icon).toHaveAttribute(
      "aria-label",
      "API key stored (not verified against the provider)",
    );
  });
});
