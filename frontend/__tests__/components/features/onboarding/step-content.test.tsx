import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StepContent } from "#/components/features/onboarding/step-content";

describe("StepContent", () => {
  const mockOptions = [
    { id: "option1", label: "Option 1" },
    { id: "option2", label: "Option 2" },
    { id: "option3", label: "Option 3" },
  ];

  const defaultProps = {
    options: mockOptions,
    selectedOptionId: null,
    onSelectOption: vi.fn(),
  };

  it("should render with the correct test id", () => {
    render(<StepContent {...defaultProps} />);

    expect(screen.getByTestId("step-content")).toBeInTheDocument();
  });

  it("should render all options", () => {
    render(<StepContent {...defaultProps} />);

    expect(screen.getByText("Option 1")).toBeInTheDocument();
    expect(screen.getByText("Option 2")).toBeInTheDocument();
    expect(screen.getByText("Option 3")).toBeInTheDocument();
  });

  it("should call onSelectOption with correct id when option is clicked", async () => {
    const onSelectOptionMock = vi.fn();
    const user = userEvent.setup();

    render(
      <StepContent {...defaultProps} onSelectOption={onSelectOptionMock} />,
    );

    await user.click(screen.getByTestId("step-option-option2"));

    expect(onSelectOptionMock).toHaveBeenCalledWith("option2");
  });

  it("should mark the selected option as selected", () => {
    render(<StepContent {...defaultProps} selectedOptionId="option1" />);

    const selectedOption = screen.getByTestId("step-option-option1");
    const unselectedOption = screen.getByTestId("step-option-option2");

    expect(selectedOption).toHaveClass("border-white");
    expect(unselectedOption).toHaveClass("border-[#3a3a3a]");
  });

  it("should render no options when options array is empty", () => {
    render(<StepContent {...defaultProps} options={[]} />);

    expect(screen.getByTestId("step-content")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("should render correct number of options", () => {
    render(<StepContent {...defaultProps} />);

    const options = screen.getAllByRole("button");
    expect(options).toHaveLength(3);
  });

  it("should allow selecting different options", async () => {
    const onSelectOptionMock = vi.fn();
    const user = userEvent.setup();

    render(
      <StepContent {...defaultProps} onSelectOption={onSelectOptionMock} />,
    );

    await user.click(screen.getByTestId("step-option-option1"));
    expect(onSelectOptionMock).toHaveBeenCalledWith("option1");

    await user.click(screen.getByTestId("step-option-option3"));
    expect(onSelectOptionMock).toHaveBeenCalledWith("option3");

    expect(onSelectOptionMock).toHaveBeenCalledTimes(2);
  });
});
