import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WorkspacePath } from "#/components/features/files-tab/workspace-path";

const originalClipboard = navigator.clipboard;

describe("WorkspacePath", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: originalClipboard,
    });
  });

  it("shows the effective workspace path", () => {
    const path = "/Users/alice/workspace/project/abc123";

    render(<WorkspacePath path={path} />);

    expect(screen.getByTestId("files-tab-workspace-path")).toHaveTextContent(
      "WORKSPACE$TITLE:",
    );
    expect(
      screen.getByTestId("files-tab-workspace-path-value"),
    ).toHaveTextContent(path);
  });

  it("keeps the complete path available when the text is truncated", () => {
    const path =
      "/Users/alice/a-very-long-workspace-name/project/with/nested/directories";

    render(<WorkspacePath path={path} />);

    const value = screen.getByTestId("files-tab-workspace-path-value");
    expect(value).toHaveClass("truncate");
    expect(value).toHaveAttribute("title", path);
  });

  it("copies the complete path and confirms the action", async () => {
    const path = "C:\\Users\\alice\\workspace\\project";
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(<WorkspacePath path={path} />);

    const workspacePath = screen.getByTestId("files-tab-workspace-path");
    await user.click(
      within(workspacePath).getByRole("button", { name: "BUTTON$COPY" }),
    );

    expect(writeText).toHaveBeenCalledWith(path);
    expect(
      within(workspacePath).getByRole("button", { name: "BUTTON$COPIED" }),
    ).toBeDisabled();
  });

  it("does not render without a workspace path", () => {
    const { rerender } = render(<WorkspacePath path={null} />);

    expect(
      screen.queryByTestId("files-tab-workspace-path"),
    ).not.toBeInTheDocument();

    rerender(<WorkspacePath path="   " />);
    expect(
      screen.queryByTestId("files-tab-workspace-path"),
    ).not.toBeInTheDocument();
  });
});
