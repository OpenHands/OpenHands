import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ToolItem } from "#/components/features/conversation-panel/system-message-modal/tool-item";
import type { ChatCompletionToolParam } from "#/types/v1/core";

describe("ToolItem", () => {
  const user = userEvent.setup();
  const onToggleMock = vi.fn();

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("Name/Title Extraction", () => {
    it("should display name from V0 format function.name", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Test description",
          parameters: {},
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toggleButton = screen.getByTestId("toggle-button");
      expect(toggleButton).toHaveTextContent("test_function");
    });

    it("should display title from V1 format root level", () => {
      // Arrange
      const v1Tool = {
        title: "V1 Tool Title",
        description: "V1 description",
        parameters: { type: "object" },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toggleButton = screen.getByTestId("toggle-button");
      expect(toggleButton).toHaveTextContent("V1 Tool Title");
    });

    it("should prioritize root title over annotations.title in V1 format", () => {
      // Arrange
      const v1Tool = {
        title: "Root Title",
        annotations: {
          title: "Annotations Title",
        },
        description: "Description",
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toggleButton = screen.getByTestId("toggle-button");
      expect(toggleButton).toHaveTextContent("Root Title");
    });

    it("should fallback to annotations.title when root title is missing", () => {
      // Arrange
      const v1Tool = {
        annotations: {
          title: "Annotations Title",
        },
        description: "Description",
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toggleButton = screen.getByTestId("toggle-button");
      expect(toggleButton).toHaveTextContent("Annotations Title");
    });

    it("should display empty string when no name or title is available", () => {
      // Arrange
      const toolWithoutName = {
        description: "Description only",
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={toolWithoutName}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toggleButton = screen.getByTestId("toggle-button");
      expect(toggleButton).toHaveTextContent("");
    });
  });

  describe("Description Extraction", () => {
    it("should display description from V0 format function.description when expanded", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "V0 function description",
          parameters: {},
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const markdownRenderer = screen.getByTestId("markdown-renderer");
      expect(markdownRenderer).toHaveTextContent("V0 function description");
    });

    it("should display description from V1 format root level when expanded", () => {
      // Arrange
      const v1Tool = {
        title: "V1 Tool",
        description: "V1 root description",
        parameters: {},
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const markdownRenderer = screen.getByTestId("markdown-renderer");
      expect(markdownRenderer).toHaveTextContent("V1 root description");
    });

    it("should prioritize root description over function.description in V1 format", () => {
      // Arrange
      const v1Tool = {
        title: "V1 Tool",
        description: "Root description",
        function: {
          description: "Function description",
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const markdownRenderer = screen.getByTestId("markdown-renderer");
      expect(markdownRenderer).toHaveTextContent("Root description");
    });

    it("should display empty string when no description is available", () => {
      // Arrange
      const toolWithoutDescription = {
        title: "Tool Name",
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={toolWithoutDescription}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const markdownRenderer = screen.getByTestId("markdown-renderer");
      expect(markdownRenderer).toHaveTextContent("");
    });

    it("should not display description when collapsed", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Should not be visible",
          parameters: {},
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.queryByTestId("markdown-renderer")).not.toBeInTheDocument();
    });
  });

  describe("Parameters Extraction", () => {
    it("should display parameters from V0 format function.parameters when expanded", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Description",
          parameters: {
            type: "object",
            properties: {
              param1: { type: "string" },
            },
          },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toolParameters = screen.getByTestId("tool-parameters");
      expect(toolParameters).toBeInTheDocument();
      // Verify that the parameters are rendered (ReactJsonView will render the JSON)
      expect(toolParameters).toHaveTextContent("param1");
    });

    it("should display parameters from V1 format root level when expanded", () => {
      // Arrange
      const v1Tool = {
        title: "V1 Tool",
        description: "Description",
        parameters: {
          type: "object",
          properties: {
            param2: { type: "number" },
          },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v1Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toolParameters = screen.getByTestId("tool-parameters");
      expect(toolParameters).toBeInTheDocument();
      // Verify that the parameters are rendered (ReactJsonView will render the JSON)
      expect(toolParameters).toHaveTextContent("param2");
    });

    it("should prioritize function.parameters over root parameters in V0 format", () => {
      // Arrange
      const v0Tool = {
        type: "function",
        function: {
          name: "test_function",
          description: "Description",
          parameters: {
            type: "object",
            source: "function",
          },
        },
        parameters: {
          type: "object",
          source: "root",
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      const toolParameters = screen.getByTestId("tool-parameters");
      expect(toolParameters).toBeInTheDocument();
      // Verify that function parameters are used (not root parameters)
      expect(toolParameters).toHaveTextContent("function");
      expect(toolParameters).not.toHaveTextContent("root");
    });

    it("should not display parameters when they are null", () => {
      // Arrange
      const toolWithoutParameters = {
        title: "Tool Name",
        description: "Description",
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={toolWithoutParameters}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.queryByTestId("tool-parameters")).not.toBeInTheDocument();
    });

    it("should not display parameters when collapsed", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Description",
          parameters: {
            type: "object",
          },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.queryByTestId("tool-parameters")).not.toBeInTheDocument();
    });
  });

  describe("Toggle Functionality", () => {
    it("should call onToggle with correct index when toggle button is clicked", async () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Description",
          parameters: {},
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={2}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      const toggleButton = screen.getByTestId("toggle-button");
      await user.click(toggleButton);

      // Assert
      expect(onToggleMock).toHaveBeenCalledOnce();
      expect(onToggleMock).toHaveBeenCalledWith(2);
    });

    it("should show expanded content when isExpanded is true", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Test description",
          parameters: { type: "object" },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.getByTestId("markdown-renderer")).toBeInTheDocument();
      expect(screen.getByTestId("tool-parameters")).toBeInTheDocument();
    });

    it("should hide expanded content when isExpanded is false", () => {
      // Arrange
      const v0Tool: ChatCompletionToolParam = {
        type: "function",
        function: {
          name: "test_function",
          description: "Test description",
          parameters: { type: "object" },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={false}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.queryByTestId("markdown-renderer")).not.toBeInTheDocument();
      expect(screen.queryByTestId("tool-parameters")).not.toBeInTheDocument();
    });
  });

  describe("Complex Scenarios", () => {
    it("should handle V0 format with type field correctly", () => {
      // Arrange
      const v0Tool = {
        type: "function",
        function: {
          name: "typed_function",
          description: "Typed description",
          parameters: { type: "object" },
        },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={v0Tool}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.getByTestId("toggle-button")).toHaveTextContent(
        "typed_function",
      );
      expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(
        "Typed description",
      );
      expect(screen.getByTestId("tool-parameters")).toBeInTheDocument();
    });

    it("should handle tool data where function is at root level (fallback behavior)", () => {
      // Arrange
      const toolWithFunctionAtRoot = {
        name: "root_function",
        description: "Root function description",
        parameters: { type: "object" },
      };

      // Act
      renderWithProviders(
        <ToolItem
          tool={toolWithFunctionAtRoot}
          index={0}
          isExpanded={true}
          onToggle={onToggleMock}
        />,
      );

      // Assert
      expect(screen.getByTestId("toggle-button")).toHaveTextContent(
        "root_function",
      );
      expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(
        "Root function description",
      );
      expect(screen.getByTestId("tool-parameters")).toBeInTheDocument();
    });
  });
});
