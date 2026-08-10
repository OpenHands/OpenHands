import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { fileEditorVisualizer } from "#/components/features/chat/tool-visualizers/file-editor/file-editor";
import {
  renderVisualizer,
  fileEditorAction,
  fileEditorObservation,
} from "../test-utils";

const Body = fileEditorVisualizer.Body;

describe("fileEditorVisualizer", () => {
  it("shows path and content for a create action", () => {
    const { container } = renderVisualizer(
      <Body
        action={fileEditorAction({
          command: "create",
          path: "/workspace/app.ts",
          file_text: "const x = 1;",
        })}
      />,
    );
    expect(container).toHaveTextContent("/workspace/app.ts");
    expect(container).toHaveTextContent("const x = 1;");
  });

  it("shows the path with a line range for a view action", () => {
    const { container } = renderVisualizer(
      <Body
        action={fileEditorAction({
          command: "view",
          path: "/workspace/app.ts",
          view_range: [1, 10],
        })}
      />,
    );
    expect(container).toHaveTextContent("/workspace/app.ts:1-10");
  });

  it("shows the file snippet the agent saw for a view observation", () => {
    const { container } = renderVisualizer(
      <Body
        observation={fileEditorObservation({
          command: "view",
          content: [
            {
              type: "text",
              text: "Here's the result of running `cat -n`:\n     1\tconst x = 1;",
            },
          ],
        })}
      />,
    );
    expect(container).toHaveTextContent("const x = 1;");
  });

  it("renders a diff for an edit observation", () => {
    const { container } = renderVisualizer(
      <Body
        observation={fileEditorObservation({
          command: "str_replace",
          old_content: "line one\nOLD\nline three",
          new_content: "line one\nNEW\nline three",
        })}
      />,
    );
    expect(container).toHaveTextContent("- OLD");
    expect(container).toHaveTextContent("+ NEW");
  });

  it("renders a diff when clearing a file (new_content is an empty string)", () => {
    const { container } = renderVisualizer(
      <Body
        observation={fileEditorObservation({
          command: "str_replace",
          old_content: "keep\nremove me",
          new_content: "",
        })}
      />,
    );
    // The empty `new_content` must not short-circuit the diff to the fallback.
    expect(container).toHaveTextContent("- keep");
    expect(container).toHaveTextContent("- remove me");
  });

  it("renders a diff when inserting into an empty file (old_content is an empty string)", () => {
    const { container } = renderVisualizer(
      <Body
        observation={fileEditorObservation({
          command: "insert",
          old_content: "",
          new_content: "first line\nsecond line",
        })}
      />,
    );
    expect(container).toHaveTextContent("+ first line");
    expect(container).toHaveTextContent("+ second line");
  });

  it("renders the inserted text for an in-flight insert action (no old_str)", () => {
    const { container } = renderVisualizer(
      <Body
        action={fileEditorAction({
          command: "insert",
          path: "/workspace/app.ts",
          new_str: "inserted line",
          insert_line: 3,
        })}
      />,
    );
    // Inserts carry `new_str` only; the card must show it, not just the path.
    expect(container).toHaveTextContent("/workspace/app.ts");
    expect(container).toHaveTextContent("+ inserted line");
  });

  it("renders a diff for an in-flight str_replace action", () => {
    const { container } = renderVisualizer(
      <Body
        action={fileEditorAction({
          command: "str_replace",
          path: "/workspace/app.ts",
          old_str: "OLD",
          new_str: "NEW",
        })}
      />,
    );
    expect(container).toHaveTextContent("- OLD");
    expect(container).toHaveTextContent("+ NEW");
  });

  it("renders the error message for a failed edit (error state)", () => {
    renderVisualizer(
      <Body
        observation={fileEditorObservation({
          command: "str_replace",
          error: "No replacement performed",
        })}
      />,
    );
    expect(screen.getByText("No replacement performed")).toBeInTheDocument();
  });
});
