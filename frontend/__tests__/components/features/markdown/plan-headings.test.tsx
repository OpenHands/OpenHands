import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { planHeadings } from "#/components/features/markdown/plan-headings";

describe("planHeadings", () => {
  describe("h1", () => {
    it("should render h1 with correct text content", () => {
      // Arrange
      const H1 = planHeadings.h1;
      const text = "Main Heading";

      // Act
      render(<H1>{text}</H1>);

      // Assert
      const heading = screen.getByRole("heading", { level: 1 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });

    it("should handle undefined children gracefully", () => {
      // Arrange
      const H1 = planHeadings.h1;

      // Act
      render(<H1>{undefined}</H1>);

      // Assert
      const heading = screen.getByRole("heading", { level: 1 });
      expect(heading).toBeInTheDocument();
      expect(heading).toBeEmptyDOMElement();
    });

    it("should render complex children content", () => {
      // Arrange
      const H1 = planHeadings.h1;

      // Act
      render(
        <H1>
          <span>Nested</span> Content
        </H1>,
      );

      // Assert
      const heading = screen.getByRole("heading", { level: 1 });
      expect(heading).toHaveTextContent("Nested Content");
      expect(heading.querySelector("span")).toHaveTextContent("Nested");
    });
  });

  describe("h2", () => {
    it("should render h2 with correct text content", () => {
      // Arrange
      const H2 = planHeadings.h2;
      const text = "Section Heading";

      // Act
      render(<H2>{text}</H2>);

      // Assert
      const heading = screen.getByRole("heading", { level: 2 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });

    it("should handle null children gracefully", () => {
      // Arrange
      const H2 = planHeadings.h2;

      // Act
      render(<H2>{null}</H2>);

      // Assert
      const heading = screen.getByRole("heading", { level: 2 });
      expect(heading).toBeInTheDocument();
      expect(heading).toBeEmptyDOMElement();
    });
  });

  describe("h3", () => {
    it("should render h3 with correct text content", () => {
      // Arrange
      const H3 = planHeadings.h3;
      const text = "Subsection Heading";

      // Act
      render(<H3>{text}</H3>);

      // Assert
      const heading = screen.getByRole("heading", { level: 3 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });
  });

  describe("h4", () => {
    it("should render h4 with correct text content", () => {
      // Arrange
      const H4 = planHeadings.h4;
      const text = "Level 4 Heading";

      // Act
      render(<H4>{text}</H4>);

      // Assert
      const heading = screen.getByRole("heading", { level: 4 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });
  });

  describe("h5", () => {
    it("should render h5 with correct text content", () => {
      // Arrange
      const H5 = planHeadings.h5;
      const text = "Level 5 Heading";

      // Act
      render(<H5>{text}</H5>);

      // Assert
      const heading = screen.getByRole("heading", { level: 5 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });
  });

  describe("h6", () => {
    it("should render h6 with correct text content", () => {
      // Arrange
      const H6 = planHeadings.h6;
      const text = "Level 6 Heading";

      // Act
      render(<H6>{text}</H6>);

      // Assert
      const heading = screen.getByRole("heading", { level: 6 });
      expect(heading).toBeInTheDocument();
      expect(heading).toHaveTextContent(text);
    });
  });

  describe("heading hierarchy", () => {
    it("should render all heading levels correctly in sequence", () => {
      // Arrange
      const H1 = planHeadings.h1;
      const H2 = planHeadings.h2;
      const H3 = planHeadings.h3;
      const H4 = planHeadings.h4;
      const H5 = planHeadings.h5;
      const H6 = planHeadings.h6;

      // Act
      render(
        <div>
          <H1>Heading 1</H1>
          <H2>Heading 2</H2>
          <H3>Heading 3</H3>
          <H4>Heading 4</H4>
          <H5>Heading 5</H5>
          <H6>Heading 6</H6>
        </div>,
      );

      // Assert
      expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
        "Heading 1",
      );
      expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
        "Heading 2",
      );
      expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent(
        "Heading 3",
      );
      expect(screen.getByRole("heading", { level: 4 })).toHaveTextContent(
        "Heading 4",
      );
      expect(screen.getByRole("heading", { level: 5 })).toHaveTextContent(
        "Heading 5",
      );
      expect(screen.getByRole("heading", { level: 6 })).toHaveTextContent(
        "Heading 6",
      );
    });
  });
});
