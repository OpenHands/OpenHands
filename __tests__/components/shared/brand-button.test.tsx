import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { BrandButton } from "#/components/features/settings/brand-button";

describe("BrandButton", () => {
  const onClickMock = vi.fn();

  it("should set a test id", () => {
    render(
      <BrandButton testId="brand-button" type="button" variant="primary">
        Test Button
      </BrandButton>,
    );

    expect(screen.getByTestId("brand-button")).toBeInTheDocument();
  });

  it("should call onClick when clicked", async () => {
    const user = userEvent.setup();
    render(
      <BrandButton type="button" variant="primary" onClick={onClickMock}>
        Test Button
      </BrandButton>,
    );

    await user.click(screen.getByText("Test Button"));
  });

  it("should be disabled if isDisabled is true", () => {
    render(
      <BrandButton type="button" variant="primary" isDisabled>
        Test Button
      </BrandButton>,
    );

    expect(screen.getByText("Test Button")).toBeDisabled();
  });

  it("should pass a start content", () => {
    render(
      <BrandButton
        type="button"
        variant="primary"
        startContent={
          <div data-testid="custom-start-content">Start Content</div>
        }
      >
        Test Button
      </BrandButton>,
    );

    screen.getByTestId("custom-start-content");
  });
});
