import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FilePathChip } from "#/components/features/chat/tool-visualizers/primitives/file-path-chip";

const openWorkspaceFile = vi.fn();

vi.mock("#/services/canvas-ui", () => ({
  openWorkspaceFile: (...args: unknown[]) => openWorkspaceFile(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

describe("FilePathChip", () => {
  beforeEach(() => {
    openWorkspaceFile.mockClear();
  });

  it("opens the workspace file when clicked without a custom handler", async () => {
    const user = userEvent.setup();
    render(<FilePathChip path="test.md" />);

    await user.click(screen.getByTestId("file-path-chip"));

    expect(openWorkspaceFile).toHaveBeenCalledWith("test.md", "conv-1");
  });

  it("prefers a custom onClick over the default open handler", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<FilePathChip path="test.md" onClick={onClick} />);

    await user.click(screen.getByTestId("file-path-chip"));

    expect(onClick).toHaveBeenCalledTimes(1);
    expect(openWorkspaceFile).not.toHaveBeenCalled();
  });
});
