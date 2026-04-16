import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { EnterpriseCard } from "#/components/features/onboarding/enterprise-card";

describe("EnterpriseCard", () => {
  const defaultProps = {
    icon: <svg data-testid="test-icon" />,
    title: "Test Title",
    description: "Test description",
    features: ["Feature 1", "Feature 2"],
    learnMoreLabel: "Learn More",
    onLearnMore: vi.fn(),
  };

  const renderWithRouter = (props = defaultProps) =>
    render(
      <MemoryRouter>
        <EnterpriseCard {...props} />
      </MemoryRouter>,
    );

  it("should render the card with title", () => {
    renderWithRouter();

    expect(screen.getByText("Test Title")).toBeInTheDocument();
  });

  it("should render the description", () => {
    renderWithRouter();

    expect(screen.getByText("Test description")).toBeInTheDocument();
  });

  it("should render the icon", () => {
    renderWithRouter();

    expect(screen.getByTestId("test-icon")).toBeInTheDocument();
  });

  it("should render the features", () => {
    renderWithRouter();

    expect(screen.getByText("Feature 1")).toBeInTheDocument();
    expect(screen.getByText("Feature 2")).toBeInTheDocument();
  });

  it("should render the learn more link with correct label", () => {
    renderWithRouter();

    const link = screen.getByRole("link", {
      name: "Learn More Test Title",
    });
    expect(link).toBeInTheDocument();
  });

  it("should have correct href", () => {
    renderWithRouter();

    const link = screen.getByRole("link", { name: "Learn More Test Title" });
    expect(link).toHaveAttribute("href", "/information-request");
  });

  it("should call onLearnMore when link is clicked", async () => {
    const mockOnLearnMore = vi.fn();
    const user = userEvent.setup();

    renderWithRouter({ ...defaultProps, onLearnMore: mockOnLearnMore });

    const link = screen.getByRole("link", { name: "Learn More Test Title" });
    await user.click(link);

    expect(mockOnLearnMore).toHaveBeenCalledTimes(1);
  });

  it("should have correct aria-label on link", () => {
    renderWithRouter();

    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("aria-label", "Learn More Test Title");
  });
});
