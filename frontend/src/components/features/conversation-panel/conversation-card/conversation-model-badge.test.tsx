import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ConversationModelBadge } from "./conversation-model-badge";

describe("ConversationModelBadge", () => {
  it("renders nothing when llmModel is null", () => {
    const { container } = render(<ConversationModelBadge llmModel={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when llmModel is undefined", () => {
    const { container } = render(<ConversationModelBadge />);
    expect(container.firstChild).toBeNull();
  });

  it("renders badge with provider and model", () => {
    render(<ConversationModelBadge llmModel="openai/gpt-4" />);
    expect(screen.getByText("OpenAI/gpt-4")).toBeInTheDocument();
  });

  it("renders badge with openhands provider", () => {
    render(
      <ConversationModelBadge llmModel="openhands/claude-opus-4-5-20251101" />,
    );
    expect(
      screen.getByText("OpenHands/claude-opus-4-5-20251101"),
    ).toBeInTheDocument();
  });

  it("renders model only when no provider is found", () => {
    render(<ConversationModelBadge llmModel="unknown-model" />);
    expect(screen.getByText("unknown-model")).toBeInTheDocument();
  });

  it("handles anthropic models correctly", () => {
    render(<ConversationModelBadge llmModel="anthropic/claude-3-opus" />);
    expect(screen.getByText("Anthropic/claude-3-opus")).toBeInTheDocument();
  });

  it("handles azure models correctly", () => {
    render(<ConversationModelBadge llmModel="azure/gpt-4" />);
    expect(screen.getByText("Azure/gpt-4")).toBeInTheDocument();
  });
});
