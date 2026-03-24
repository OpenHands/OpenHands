import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Card } from "#/ui/card";

describe("Card", () => {
  it("should render children", () => {
    render(<Card>Card Content</Card>);

    expect(screen.getByText("Card Content")).toBeInTheDocument();
  });

  it("should render with testId", () => {
    render(<Card testId="test-card">Content</Card>);

    expect(screen.getByTestId("test-card")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    render(
      <Card testId="test-card" className="custom-class">
        Content
      </Card>,
    );

    expect(screen.getByTestId("test-card")).toHaveClass("custom-class");
  });

  describe("theme variants", () => {
    it("should apply default theme styles", () => {
      render(<Card testId="test-card">Content</Card>);

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("bg-[#26282D]");
      expect(card).toHaveClass("border-[#727987]");
      expect(card).toHaveClass("rounded-xl");
    });

    it("should apply outlined theme styles", () => {
      render(
        <Card testId="test-card" theme="outlined">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("bg-transparent");
      expect(card).toHaveClass("border-[#727987]");
    });

    it("should apply dark theme styles", () => {
      render(
        <Card testId="test-card" theme="dark">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("bg-black");
      expect(card).toHaveClass("border-[#242424]");
      expect(card).toHaveClass("rounded-2xl");
    });
  });

  describe("hover variants", () => {
    it("should not apply hover styles when hover is none", () => {
      render(
        <Card testId="test-card" hover="none">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).not.toHaveClass("hover:bg-[linear-gradient");
    });

    it("should apply elevated hover styles", () => {
      render(
        <Card testId="test-card" hover="elevated">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("transition-all");
      expect(card).toHaveClass("duration-200");
    });
  });

  describe("gradient variants", () => {
    it("should not apply gradient styles when gradient is none", () => {
      render(
        <Card testId="test-card" gradient="none">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).not.toHaveClass("bg-[#0A0A0A80]");
    });

    it("should apply standard gradient styles", () => {
      render(
        <Card testId="test-card" gradient="standard">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("bg-[#0A0A0A80]");
      expect(card).toHaveClass("border-t-[#24242499]");
    });
  });

  describe("combined variants", () => {
    it("should apply dark theme with standard gradient", () => {
      render(
        <Card testId="test-card" theme="dark" gradient="standard">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      // Should have dark theme base
      expect(card).toHaveClass("border-[#242424]");
      expect(card).toHaveClass("rounded-2xl");
      // Should have gradient overlay
      expect(card).toHaveClass("bg-[#0A0A0A80]");
    });

    it("should apply dark theme with elevated hover", () => {
      render(
        <Card testId="test-card" theme="dark" hover="elevated">
          Content
        </Card>,
      );

      const card = screen.getByTestId("test-card");
      expect(card).toHaveClass("rounded-2xl");
      expect(card).toHaveClass("transition-all");
    });
  });

  it("should have flex display by default", () => {
    render(<Card testId="test-card">Content</Card>);

    expect(screen.getByTestId("test-card")).toHaveClass("flex");
  });

  it("should have relative positioning", () => {
    render(<Card testId="test-card">Content</Card>);

    expect(screen.getByTestId("test-card")).toHaveClass("relative");
  });
});
