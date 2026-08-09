import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AgentCanvasUpdateCard } from "./agent-canvas-update-card";

describe("AgentCanvasUpdateCard", () => {
  it("renders nothing (Heimdall fork disables upstream version UI)", () => {
    render(<AgentCanvasUpdateCard />);

    expect(
      screen.queryByTestId("agent-canvas-update-card"),
    ).not.toBeInTheDocument();
  });
});
