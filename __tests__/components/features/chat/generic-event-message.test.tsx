import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
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
});
