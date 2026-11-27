import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  planComponents,
  createPlanComponents,
} from "#/components/features/markdown/plan-components";

describe("planComponents", () => {
  describe("h1", () => {
    it("should render h1 with correct text content", () => {
      // Arrange
      const H1 = planComponents.h1;
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
      const H1 = planComponents.h1;

      // Act
      render(<H1>{undefined}</H1>);

      // Assert
      const heading = screen.getByRole("heading", { level: 1 });
      expect(heading).toBeInTheDocument();
      expect(heading).toBeEmptyDOMElement();
    });

    it("should render complex children content", () => {
      // Arrange
      const H1 = planComponents.h1;

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
      const H2 = planComponents.h2;
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
      const H2 = planComponents.h2;

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
      const H3 = planComponents.h3;
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
      const H4 = planComponents.h4;
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
      const H5 = planComponents.h5;
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
      const H6 = planComponents.h6;
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
      const H1 = planComponents.h1;
      const H2 = planComponents.h2;
      const H3 = planComponents.h3;
      const H4 = planComponents.h4;
      const H5 = planComponents.h5;
      const H6 = planComponents.h6;

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

  describe("p", () => {
    it("should render paragraph with correct text content", () => {
      // Arrange
      const P = planComponents.p;
      const text = "Paragraph text";

      // Act
      render(<P>{text}</P>);

      // Assert
      expect(screen.getByText(text)).toBeInTheDocument();
    });
  });

  describe("ul", () => {
    it("should render unordered list with correct content", () => {
      // Arrange
      const Ul = planComponents.ul;
      const Li = planComponents.li;

      // Act
      render(
        <Ul>
          <Li>Item 1</Li>
          <Li>Item 2</Li>
        </Ul>,
      );

      // Assert
      expect(screen.getByRole("list")).toBeInTheDocument();
      expect(screen.getAllByRole("listitem")).toHaveLength(2);
    });
  });

  describe("ol", () => {
    it("should render ordered list with correct content", () => {
      // Arrange
      const Ol = planComponents.ol;
      const Li = planComponents.li;

      // Act
      render(
        <Ol>
          <Li>Item 1</Li>
          <Li>Item 2</Li>
        </Ol>,
      );

      // Assert
      expect(screen.getByRole("list")).toBeInTheDocument();
      expect(screen.getAllByRole("listitem")).toHaveLength(2);
    });

    it("should support start attribute", () => {
      // Arrange
      const Ol = planComponents.ol;
      const Li = planComponents.li;

      // Act
      const { container } = render(
        <Ol start={5}>
          <Li>Item 1</Li>
        </Ol>,
      );

      // Assert
      const ol = container.querySelector("ol");
      expect(ol).toHaveAttribute("start", "5");
    });
  });

  describe("a", () => {
    it("should render anchor with correct href and content", () => {
      // Arrange
      const A = planComponents.a;
      const href = "https://example.com";
      const text = "Link text";

      // Act
      render(<A href={href}>{text}</A>);

      // Assert
      const link = screen.getByRole("link");
      expect(link).toHaveTextContent(text);
      expect(link).toHaveAttribute("href", href);
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    });
  });

  describe("code", () => {
    it("should render inline code with correct content", () => {
      // Arrange
      const Code = planComponents.code;
      const text = "const x = 1";

      // Act
      render(<Code>{text}</Code>);

      // Assert
      expect(screen.getByText(text)).toBeInTheDocument();
    });
  });
});

describe("createPlanComponents", () => {
  it("should apply extraClassName to all elements when provided", () => {
    // Arrange
    const extraClassName = "test-extra-class";
    const components = createPlanComponents(extraClassName);

    // Act
    const { container } = render(
      <div>
        <components.h1>H1</components.h1>
        <components.p>Paragraph</components.p>
        <components.ul>
          <components.li>Item</components.li>
        </components.ul>
        <components.a href="#">Link</components.a>
        <components.code>Code</components.code>
      </div>,
    );

    // Assert
    expect(container.querySelector("h1")).toHaveClass(extraClassName);
    expect(container.querySelector("p")).toHaveClass(extraClassName);
    expect(container.querySelector("ul")).toHaveClass(extraClassName);
    expect(container.querySelector("li")).toHaveClass(extraClassName);
    expect(container.querySelector("a")).toHaveClass(extraClassName);
    expect(container.querySelector("code")).toHaveClass(extraClassName);
  });

  it("should not add undefined class when extraClassName is not provided", () => {
    // Arrange
    const components = createPlanComponents();

    // Act
    const { container } = render(<components.h1>H1</components.h1>);

    // Assert
    const h1 = container.querySelector("h1");
    expect(h1?.className).not.toContain("undefined");
  });
});
