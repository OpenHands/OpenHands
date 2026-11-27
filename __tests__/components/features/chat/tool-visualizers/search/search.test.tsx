import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { searchVisualizer } from "#/components/features/chat/tool-visualizers/search/search";
import {
  renderVisualizer,
  grepAction,
  grepObservation,
  globObservation,
} from "../test-utils";

const Body = searchVisualizer.Body;

describe("searchVisualizer", () => {
  it("shows search params on the action card", () => {
    const { container } = renderVisualizer(
      <Body
        action={grepAction({ pattern: "TODO", path: "/src", include: "*.ts" })}
      />,
    );
    expect(screen.getByText("COMMON$PATTERN")).toBeInTheDocument();
    expect(container).toHaveTextContent("TODO");
    expect(container).toHaveTextContent("*.ts");
  });

  it("lists matches with a count for a grep observation", () => {
    renderVisualizer(
      <Body
        observation={grepObservation({
          pattern: "TODO",
          matches: ["a.ts", "b.ts"],
        })}
      />,
    );
    expect(screen.getByText("COMMON$RESULTS")).toBeInTheDocument();
    expect(screen.getByText("a.ts")).toBeInTheDocument();
    expect(screen.getByText("b.ts")).toBeInTheDocument();
  });

  it("shows a no-results message when empty", () => {
    renderVisualizer(
      <Body observation={grepObservation({ pattern: "TODO" })} />,
    );
    expect(screen.getByText("COMMON$NO_RESULTS")).toBeInTheDocument();
  });

  it("renders the error text when the search failed (error state)", () => {
    renderVisualizer(
      <Body
        observation={grepObservation({
          pattern: "(",
          is_error: true,
          content: [{ type: "text", text: "invalid regex" }],
        })}
      />,
    );
    expect(screen.getByText("invalid regex")).toBeInTheDocument();
  });

  it("flags truncated glob results", () => {
    renderVisualizer(
      <Body
        observation={globObservation({
          pattern: "**/*.ts",
          files: ["a.ts"],
          truncated: true,
        })}
      />,
    );
    expect(screen.getByText("COMMON$TRUNCATED")).toBeInTheDocument();
  });
});
