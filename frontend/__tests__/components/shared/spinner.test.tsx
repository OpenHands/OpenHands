import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { Spinner } from "#/components/shared/spinner";

describe("Spinner", () => {
  it("should render with default props", () => {
    // Arrange & Act
    render(<Spinner />);

    // Assert
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toBeInTheDocument();
  });

  it("should render with label", () => {
    // Arrange
    const label = "Loading...";

    // Act
    render(<Spinner label={label} />);

    // Assert
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    // Arrange
    const customClass = "custom-class";

    // Act
    render(<Spinner className={customClass} />);

    // Assert
    const spinner = screen.getByTestId("spinner");
    expect(spinner).toHaveClass(customClass);
  });

  it("should render with different sizes", () => {
    // Arrange
    const sizes = ["sm", "md", "lg", "xl"] as const;

    // Act & Assert
    sizes.forEach((size) => {
      const { container } = render(<Spinner size={size} />);
      const spinner = screen.getByTestId("spinner");
      expect(spinner).toBeInTheDocument();
      
      // Clean up for next iteration
      container.remove();
    });
  });

  it("should apply custom dataTestId", () => {
    // Arrange
    const testId = "custom-spinner";

    // Act
    render(<Spinner dataTestId={testId} />);

    // Assert
    expect(screen.getByTestId(testId)).toBeInTheDocument();
  });
});

