import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SecurityRisk } from "#/types/agent-server/core";
import { bashVisualizer } from "#/components/features/chat/tool-visualizers/bash/bash";
import { MAX_CONTENT_LENGTH } from "#/components/conversation-events/chat/event-content-helpers/shared";
import {
  renderVisualizer,
  bashAction,
  bashObservation,
  terminalObservation,
} from "../test-utils";

const Body = bashVisualizer.Body;

describe("bashVisualizer", () => {
  it("renders the command in the action card", () => {
    const { container } = renderVisualizer(
      <Body action={bashAction("echo hello")} />,
    );
    expect(container).toHaveTextContent("echo hello");
  });

  it("warns about high-risk actions", () => {
    renderVisualizer(
      <Body action={bashAction("rm -rf /", SecurityRisk.HIGH)} />,
    );
    expect(screen.getByText("SECURITY$HIGH_RISK")).toBeInTheDocument();
  });

  it("shows output without an exit badge on success", () => {
    const { container } = renderVisualizer(
      <Body
        observation={bashObservation("hello world", 0, "echo hello world")}
      />,
    );
    expect(container).toHaveTextContent("hello world");
    expect(screen.queryByText("OBSERVATION$EXIT_CODE")).not.toBeInTheDocument();
  });

  it("badges a non-zero exit code (error state)", () => {
    renderVisualizer(<Body observation={bashObservation("boom", 1)} />);
    expect(screen.getByText("OBSERVATION$EXIT_CODE")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("shows a placeholder when there is no output", () => {
    const { container } = renderVisualizer(
      <Body observation={bashObservation("", 0)} />,
    );
    expect(container).toHaveTextContent("OBSERVATION$COMMAND_NO_OUTPUT");
  });

  it("exposes a copy button for the command and the output", () => {
    renderVisualizer(
      <Body
        observation={bashObservation("hello world", 0, "echo hello world")}
      />,
    );
    // One for the command code block, one for the output pane.
    expect(screen.getAllByTestId("copy-to-clipboard")).toHaveLength(2);
  });

  it("has no copy button when there is no output to copy", () => {
    renderVisualizer(<Body observation={bashObservation("", 0, "true")} />);
    // Only the command's copy button remains.
    expect(screen.getAllByTestId("copy-to-clipboard")).toHaveLength(1);
  });

  it("expands long command output inline", async () => {
    const user = userEvent.setup();
    const longOutput = `${"a".repeat(MAX_CONTENT_LENGTH)}tail-marker`;
    const { container } = renderVisualizer(
      <Body observation={bashObservation(longOutput, 0)} />,
    );

    expect(container).not.toHaveTextContent("tail-marker");

    await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));

    expect(container).toHaveTextContent("tail-marker");
    expect(
      screen.getByRole("button", { name: "BUTTON$COLLAPSE" }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "BUTTON$COLLAPSE" }));

    expect(container).not.toHaveTextContent("tail-marker");
  });

  it("renders command and output for the terminal tool", () => {
    const { container } = renderVisualizer(
      <Body observation={terminalObservation("362 index.html", 0, "wc -l")} />,
    );
    expect(container).toHaveTextContent("wc -l");
    expect(container).toHaveTextContent("362 index.html");
  });
});
