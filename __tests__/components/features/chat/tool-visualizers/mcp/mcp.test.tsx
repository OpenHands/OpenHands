import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { mcpVisualizer } from "#/components/features/chat/tool-visualizers/mcp/mcp";
import { renderVisualizer, mcpObservation } from "../test-utils";

const Body = mcpVisualizer.Body;

describe("mcpVisualizer", () => {
  it("names the MCP tool that ran", () => {
    renderVisualizer(
      <Body observation={mcpObservation({ tool_name: "start_server" })} />,
    );
    expect(screen.getByText("start_server")).toBeInTheDocument();
  });

  it("shows the output the agent received", () => {
    const { container } = renderVisualizer(
      <Body
        observation={mcpObservation({
          content: [{ type: "text", text: "server started on :8080" }],
        })}
      />,
    );
    expect(container).toHaveTextContent("server started on :8080");
  });

  it("surfaces the error text in the error style when the call failed", () => {
    const { container } = renderVisualizer(
      <Body
        observation={mcpObservation({
          content: [{ type: "text", text: "connection refused" }],
          is_error: true,
        })}
      />,
    );
    const error = container.querySelector(".text-danger");
    expect(error).toHaveTextContent("connection refused");
  });

  it("shows no exit-code badge, since MCP observations carry none", () => {
    renderVisualizer(<Body observation={mcpObservation()} />);
    expect(screen.queryByText("OBSERVATION$EXIT_CODE")).not.toBeInTheDocument();
  });

  it("renders no process details before the observation arrives", () => {
    const { container } = renderVisualizer(<Body />);
    expect(container).toHaveTextContent("");
    expect(screen.queryByText("OBSERVATION$TOOL")).not.toBeInTheDocument();
    expect(screen.queryByText("OBSERVATION$EXIT_CODE")).not.toBeInTheDocument();
  });
});
