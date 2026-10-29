import { afterEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GenericEventMessage } from "#/components/features/chat/generic-event-message";

describe("GenericEventMessage", () => {
  it("opens and closes the details when the title text is clicked", async () => {
    const user = userEvent.setup();
    render(<GenericEventMessage title="Ran tests" details="test output" />);

    expect(screen.queryByText("test output")).not.toBeInTheDocument();
    await user.click(screen.getByText("Ran tests"));
    expect(screen.getByText("test output")).toBeInTheDocument();
    await user.click(screen.getByText("Ran tests"));
    expect(screen.queryByText("test output")).not.toBeInTheDocument();
  });

  it("toggles once when the caret button is clicked", async () => {
    const user = userEvent.setup();
    render(<GenericEventMessage title="Ran tests" details="test output" />);

    await user.click(screen.getByRole("button"));
    expect(screen.getByText("test output")).toBeInTheDocument();
  });

  it("leaves links inside the title to do their own thing", async () => {
    const user = userEvent.setup();
    render(
      <GenericEventMessage
        title={
          <a href="#file" onClick={(event) => event.preventDefault()}>
            README.md
          </a>
        }
        details="file contents"
      />,
    );

    await user.click(screen.getByRole("link", { name: "README.md" }));
    expect(screen.queryByText("file contents")).not.toBeInTheDocument();
  });

  it("is not clickable when there are no details", () => {
    render(<GenericEventMessage title="Finished" details="" />);

    expect(screen.getByTestId("generic-event-message-title")).not.toHaveClass(
      "cursor-pointer",
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  describe("timestamp tooltip", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    const renderWithTimestamp = () =>
      render(
        <GenericEventMessage
          title="Ran tests"
          details="test output"
          timestamp="2026-04-16T19:32:29.828Z"
        />,
      );

    it("waits for a resting pointer before showing", () => {
      vi.useFakeTimers();
      renderWithTimestamp();

      fireEvent.mouseEnter(screen.getByTestId("generic-event-message-title"));
      act(() => {
        vi.advanceTimersByTime(300);
      });
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(500);
      });
      expect(screen.getByRole("tooltip")).toBeInTheDocument();
    });

    it("never appears for a quick click on the title", () => {
      vi.useFakeTimers();
      renderWithTimestamp();
      const title = screen.getByTestId("generic-event-message-title");

      fireEvent.mouseEnter(title);
      fireEvent.click(screen.getByText("Ran tests"));
      act(() => {
        vi.advanceTimersByTime(1000);
      });

      expect(screen.getByText("test output")).toBeInTheDocument();
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    });
  });
});
