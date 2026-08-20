import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { SettingsInput } from "#/components/features/settings/settings-input";

describe("SettingsInput", () => {
  it("should render an optional tag if showOptionalTag is true", async () => {
    const { rerender } = render(
      <SettingsInput testId="test-input" label="Test Input" type="text" />,
    );

    expect(screen.queryByText(/optional/i)).not.toBeInTheDocument();

    rerender(
      <SettingsInput
        testId="test-input"
        showOptionalTag
        label="Test Input"
        type="text"
      />,
    );

    expect(screen.getByText(/optional/i)).toBeInTheDocument();
  });

  it("should disable the input if isDisabled is true", async () => {
    const { rerender } = render(
      <SettingsInput testId="test-input" label="Test Input" type="text" />,
    );

    expect(screen.getByTestId("test-input")).toBeEnabled();

    rerender(
      <SettingsInput
        testId="test-input"
        label="Test Input"
        type="text"
        isDisabled
      />,
    );

    expect(screen.getByTestId("test-input")).toBeDisabled();
  });

  it("should set a placeholder on the input", async () => {
    render(
      <SettingsInput
        testId="test-input"
        label="Test Input"
        type="text"
        placeholder="Test Placeholder"
      />,
    );

    expect(screen.getByTestId("test-input")).toHaveAttribute(
      "placeholder",
      "Test Placeholder",
    );
  });

  it("should set a default value on the input", async () => {
    render(
      <SettingsInput
        testId="test-input"
        label="Test Input"
        type="text"
        defaultValue="Test Value"
      />,
    );

    expect(screen.getByTestId("test-input")).toHaveValue("Test Value");
  });

  it("should render start content", async () => {
    const startContent = <div>Start Content</div>;

    render(
      <SettingsInput
        testId="test-input"
        label="Test Input"
        type="text"
        defaultValue="Test Value"
        startContent={startContent}
      />,
    );

    expect(screen.getByText("Start Content")).toBeInTheDocument();
  });

  it("should call onChange with the input value", async () => {
    const onChangeMock = vi.fn();
    const user = userEvent.setup();

    render(
      <SettingsInput
        testId="test-input"
        label="Test Input"
        type="text"
        onChange={onChangeMock}
      />,
    );

    const input = screen.getByTestId("test-input");
    await user.type(input, "Test");

    expect(onChangeMock).toHaveBeenCalledTimes(4);
    expect(onChangeMock).toHaveBeenNthCalledWith(4, "Test");
  });
});
