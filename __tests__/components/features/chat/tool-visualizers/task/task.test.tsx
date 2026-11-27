import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { taskVisualizer } from "#/components/features/chat/tool-visualizers/task/task";
import { renderVisualizer, taskAction, taskObservation } from "../test-utils";

const Body = taskVisualizer.Body;

describe("taskVisualizer", () => {
  it("shows the subagent and query while the observation is pending", () => {
    renderVisualizer(
      <Body
        action={taskAction({
          subagent_type: "code-explorer",
          prompt: "Summarize the README",
        })}
      />,
    );
    // Subagent comes from the action; no task id / result yet.
    expect(screen.getByText("TASK$SUBAGENT")).toBeInTheDocument();
    expect(screen.getByText("code-explorer")).toBeInTheDocument();
    expect(screen.getByText("TASK$QUERY")).toBeInTheDocument();
    expect(screen.getByText("Summarize the README")).toBeInTheDocument();
    expect(screen.queryByText("TASK$TASK_ID")).not.toBeInTheDocument();
    expect(screen.queryByText("TASK$RESULT")).not.toBeInTheDocument();
  });

  it("keeps showing the query alongside the result once both are present", () => {
    renderVisualizer(
      <Body
        action={taskAction({ prompt: "Summarize the README" })}
        observation={taskObservation({
          content: [{ type: "text", text: "All done." }],
        })}
      />,
    );
    expect(screen.getByText("TASK$QUERY")).toBeInTheDocument();
    expect(screen.getByText("Summarize the README")).toBeInTheDocument();
    expect(screen.getByText("TASK$RESULT")).toBeInTheDocument();
    expect(screen.getByText("All done.")).toBeInTheDocument();
  });

  it("shows the subagent and task id", () => {
    renderVisualizer(
      <Body
        observation={taskObservation({
          subagent: "code-explorer",
          task_id: "task_00000001",
        })}
      />,
    );
    expect(screen.getByText("TASK$SUBAGENT")).toBeInTheDocument();
    expect(screen.getByText("code-explorer")).toBeInTheDocument();
    expect(screen.getByText("TASK$TASK_ID")).toBeInTheDocument();
    expect(screen.getByText("task_00000001")).toBeInTheDocument();
  });

  it("renders the answer as formatted markdown", () => {
    const { container } = renderVisualizer(
      <Body
        observation={taskObservation({
          content: [
            {
              type: "text",
              text: "## README Summary\n\nLicensed under Apache.",
            },
          ],
        })}
      />,
    );
    expect(screen.getByText("TASK$RESULT")).toBeInTheDocument();
    // Markdown headings render as real heading elements, not raw "## ..." text.
    const heading = container.querySelector("h2");
    expect(heading).toHaveTextContent("README Summary");
    expect(container).toHaveTextContent("Licensed under Apache.");
  });

  it("omits the result section when there is no text content", () => {
    renderVisualizer(<Body observation={taskObservation({ content: [] })} />);
    expect(screen.queryByText("TASK$RESULT")).not.toBeInTheDocument();
  });

  it("exposes a copy button for the result", () => {
    renderVisualizer(
      <Body
        observation={taskObservation({
          content: [{ type: "text", text: "the answer" }],
        })}
      />,
    );
    expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
  });
});
