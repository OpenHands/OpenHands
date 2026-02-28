import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StepOption } from "#/components/features/onboarding/step-option";

describe("StepOption", () => {
  const defaultProps = {
    id: "test-option",
    label: "Test Label",
    selected: false,
    onClick: vi.fn(),
  };

  it("should render with the correct test id", () => {
    render(<StepOption {...defaultProps} />);

    expect(screen.getByTestId("step-option-test-option")).toBeInTheDocument();
  });

  it("should display the label", () => {
    render(<StepOption {...defaultProps} />);

    expect(screen.getByText("Test Label")).toBeInTheDocument();
  });

  it("should call onClick when clicked", async () => {
    const onClickMock = vi.fn();
    const user = userEvent.setup();

    render(<StepOption {...defaultProps} onClick={onClickMock} />);

    await user.click(screen.getByTestId("step-option-test-option"));

    expect(onClickMock).toHaveBeenCalledTimes(1);
  });

  it("should call onClick when Enter key is pressed", async () => {
    const onClickMock = vi.fn();
    const user = userEvent.setup();

    render(<StepOption {...defaultProps} onClick={onClickMock} />);

    const option = screen.getByTestId("step-option-test-option");
    option.focus();
    await user.keyboard("{Enter}");

    expect(onClickMock).toHaveBeenCalledTimes(1);
  });

  it("should call onClick when Space key is pressed", async () => {
    const onClickMock = vi.fn();
    const user = userEvent.setup();

    render(<StepOption {...defaultProps} onClick={onClickMock} />);

    const option = screen.getByTestId("step-option-test-option");
    option.focus();
    await user.keyboard(" ");

    expect(onClickMock).toHaveBeenCalledTimes(1);
  });

  it("should have role='button' for accessibility", () => {
    render(<StepOption {...defaultProps} />);

    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should be focusable with tabIndex=0", () => {
    render(<StepOption {...defaultProps} />);

    const option = screen.getByTestId("step-option-test-option");
    expect(option).toHaveAttribute("tabIndex", "0");
  });

  it("should have selected styling when selected is true", () => {
    render(<StepOption {...defaultProps} selected />);

    const option = screen.getByTestId("step-option-test-option");
    expect(option).toHaveClass("border-white");
  });

  it("should have unselected styling when selected is false", () => {
    render(<StepOption {...defaultProps} selected={false} />);

    const option = screen.getByTestId("step-option-test-option");
    expect(option).toHaveClass("border-[#3a3a3a]");
  });
});
