import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";

describe("BackNavButton", () => {
  it("renders a back button that fires onClick when clicked", () => {
    const onClick = vi.fn();
    render(
      <BackNavButton onClick={onClick} testId="back-button">
        <span />
      </BackNavButton>,
    );

    fireEvent.click(screen.getByTestId("back-button"));

    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
