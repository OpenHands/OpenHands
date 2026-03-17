import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StepInput } from "#/components/features/onboarding/step-input";

describe("StepInput", () => {
  const defaultProps = {
    id: "test-input",
    label: "Test Label",
    value: "",
    onChange: vi.fn(),
  };

  it("should render with correct test id", () => {
    render(<StepInput {...defaultProps} />);

    expect(screen.getByTestId("step-input-test-input")).toBeInTheDocument();
  });

  it("should render the label", () => {
    render(<StepInput {...defaultProps} />);

    expect(screen.getByText("Test Label")).toBeInTheDocument();
  });

  it("should display the provided value", () => {
    render(<StepInput {...defaultProps} value="Hello World" />);

    const input = screen.getByTestId("step-input-test-input");
    expect(input).toHaveValue("Hello World");
  });

  it("should call onChange when user types", async () => {
    const mockOnChange = vi.fn();
    const user = userEvent.setup();

    render(<StepInput {...defaultProps} onChange={mockOnChange} />);

    const input = screen.getByTestId("step-input-test-input");
    await user.type(input, "a");

    expect(mockOnChange).toHaveBeenCalledWith("a");
  });

  it("should call onChange with the full input value on each keystroke", async () => {
    const mockOnChange = vi.fn();
    const user = userEvent.setup();

    render(<StepInput {...defaultProps} onChange={mockOnChange} />);

    const input = screen.getByTestId("step-input-test-input");
    await user.type(input, "abc");

    expect(mockOnChange).toHaveBeenCalledTimes(3);
    expect(mockOnChange).toHaveBeenNthCalledWith(1, "a");
    expect(mockOnChange).toHaveBeenNthCalledWith(2, "b");
    expect(mockOnChange).toHaveBeenNthCalledWith(3, "c");
  });

  it("should use the id prop for data-testid", () => {
    render(<StepInput {...defaultProps} id="org_name" />);

    expect(screen.getByTestId("step-input-org_name")).toBeInTheDocument();
  });

  it("should render as a text input", () => {
    render(<StepInput {...defaultProps} />);

    const input = screen.getByTestId("step-input-test-input");
    expect(input).toHaveAttribute("type", "text");
  });
});
