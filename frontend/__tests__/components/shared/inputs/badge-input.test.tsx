import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { BadgeInput } from "#/components/shared/inputs/badge-input";

describe("BadgeInput", () => {
  it("should render the values", () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={["test", "test2"]} onChange={onChangeMock} />);

    expect(screen.getByText("test")).toBeInTheDocument();
    expect(screen.getByText("test2")).toBeInTheDocument();
  });

  it("should render the input's as a badge on space", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={["badge1"]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    expect(input).toHaveValue("");

    await userEvent.type(input, "test");
    await userEvent.type(input, " ");

    expect(onChangeMock).toHaveBeenCalledWith(["badge1", "test"]);
    expect(input).toHaveValue("");
  });

  it("should remove the badge on backspace", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={["badge1", "badge2"]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    expect(input).toHaveValue("");

    await userEvent.type(input, "{backspace}");

    expect(onChangeMock).toHaveBeenCalledWith(["badge1"]);
    expect(input).toHaveValue("");
  });

  it("should remove the badge on click", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={["badge1"]} onChange={onChangeMock} />);

    const removeButton = screen.getByTestId("remove-button");
    await userEvent.click(removeButton);

    expect(onChangeMock).toHaveBeenCalledWith([]);
  });

  it("should not create empty badges", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={[]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    expect(input).toHaveValue("");

    await userEvent.type(input, " ");
    expect(onChangeMock).not.toHaveBeenCalled();
  });

  it("should render the input as a badge on enter", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={[]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    await userEvent.type(input, "test{enter}");

    expect(onChangeMock).toHaveBeenCalledWith(["test"]);
    expect(input).toHaveValue("");
  });

  it("should render the input as a badge on comma", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={[]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    await userEvent.type(input, "test,");

    expect(onChangeMock).toHaveBeenCalledWith(["test"]);
    expect(input).toHaveValue("");
  });

  it("should commit pending input as a badge on blur", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={[]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    await userEvent.type(input, "test");
    await userEvent.tab();

    expect(onChangeMock).toHaveBeenCalledWith(["test"]);
    expect(input).toHaveValue("");
  });

  it("should split pasted text into multiple badges", async () => {
    const onChangeMock = vi.fn();
    render(<BadgeInput value={[]} onChange={onChangeMock} />);

    const input = screen.getByTestId("badge-input");
    await userEvent.click(input);
    await userEvent.paste("a@acme.org, b@acme.org\tc@acme.org");
    await userEvent.tab();

    expect(onChangeMock).toHaveBeenCalledWith([
      "a@acme.org",
      "b@acme.org",
      "c@acme.org",
    ]);
    expect(input).toHaveValue("");
  });
});
