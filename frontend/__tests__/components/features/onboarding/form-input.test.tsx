import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FormInput } from "#/components/features/onboarding/form-input";

describe("FormInput", () => {
  const defaultProps = {
    id: "test-input",
    label: "Test Label",
    value: "",
    onChange: vi.fn(),
  };

  it("should render with correct test id", () => {
    render(<FormInput {...defaultProps} />);

    expect(screen.getByTestId("form-input-test-input")).toBeInTheDocument();
  });

  it("should render the label", () => {
    render(<FormInput {...defaultProps} />);

    expect(screen.getByText("Test Label")).toBeInTheDocument();
  });

  it("should display the provided value", () => {
    render(<FormInput {...defaultProps} value="Hello World" />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveValue("Hello World");
  });

  it("should call onChange when user types", async () => {
    const mockOnChange = vi.fn();
    const user = userEvent.setup();

    render(<FormInput {...defaultProps} onChange={mockOnChange} />);

    const input = screen.getByTestId("form-input-test-input");
    await user.type(input, "a");

    expect(mockOnChange).toHaveBeenCalledWith("a");
  });

  it("should render as a text input by default", () => {
    render(<FormInput {...defaultProps} />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveAttribute("type", "text");
  });

  it("should render as an email input when type is email", () => {
    render(<FormInput {...defaultProps} type="email" />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveAttribute("type", "email");
  });

  it("should render a textarea when rows prop is provided", () => {
    render(<FormInput {...defaultProps} rows={4} />);

    const textarea = screen.getByTestId("form-input-test-input");
    expect(textarea.tagName).toBe("TEXTAREA");
    expect(textarea).toHaveAttribute("rows", "4");
  });

  it("should render placeholder text", () => {
    render(<FormInput {...defaultProps} placeholder="Enter text here" />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveAttribute("placeholder", "Enter text here");
  });

  it("should have aria-required attribute when required", () => {
    render(<FormInput {...defaultProps} required />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveAttribute("aria-required", "true");
  });

  it("should have aria-label attribute", () => {
    render(<FormInput {...defaultProps} />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toHaveAttribute("aria-label", "Test Label");
  });

  it("should have required attribute on input when required", () => {
    render(<FormInput {...defaultProps} required />);

    const input = screen.getByTestId("form-input-test-input");
    expect(input).toBeRequired();
  });

  it("should have required attribute on textarea when required", () => {
    render(<FormInput {...defaultProps} rows={4} required />);

    const textarea = screen.getByTestId("form-input-test-input");
    expect(textarea).toBeRequired();
  });

  it("should associate label with input via htmlFor", () => {
    render(<FormInput {...defaultProps} />);

    const label = screen.getByText("Test Label");
    const input = screen.getByTestId("form-input-test-input");

    expect(label).toHaveAttribute("for", "form-input-test-input");
    expect(input).toHaveAttribute("id", "form-input-test-input");
  });

  describe("error state", () => {
    it("should have aria-invalid true when showing error", () => {
      render(<FormInput {...defaultProps} required showError />);

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "true");
    });

    it("should have aria-invalid false when not showing error", () => {
      render(<FormInput {...defaultProps} required showError={false} />);

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "false");
    });

    it("should have aria-invalid false when showError is true but field has value", () => {
      render(<FormInput {...defaultProps} required showError value="filled" />);

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "false");
    });

    it("should have aria-invalid false when showError is true but field is not required", () => {
      render(<FormInput {...defaultProps} required={false} showError />);

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "false");
    });

    it("should have aria-invalid true on textarea when showError is true and empty", () => {
      render(<FormInput {...defaultProps} rows={4} required showError />);

      const textarea = screen.getByTestId("form-input-test-input");
      expect(textarea).toHaveAttribute("aria-invalid", "true");
    });

    it("should have aria-invalid true for invalid email when showError is true", () => {
      render(
        <FormInput {...defaultProps} type="email" value="invalid" showError />,
      );

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "true");
    });

    it("should have aria-invalid false for valid email when showError is true", () => {
      render(
        <FormInput
          {...defaultProps}
          type="email"
          value="test@example.com"
          showError
        />,
      );

      const input = screen.getByTestId("form-input-test-input");
      expect(input).toHaveAttribute("aria-invalid", "false");
    });
  });
});
