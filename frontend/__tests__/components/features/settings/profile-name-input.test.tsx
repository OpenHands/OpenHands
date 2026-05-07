import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ProfileNameInput } from "#/components/features/settings/profile-name-input";

describe("ProfileNameInput", () => {
  it("uses the plain Name label by default", () => {
    render(<ProfileNameInput value="" onChange={vi.fn()} />);

    expect(screen.getByText("SETTINGS$NAME")).toBeInTheDocument();
    expect(screen.queryByText(/COMMON\$OPTIONAL/)).not.toBeInTheDocument();
  });

  it("renders an Optional suffix when isOptional is true", () => {
    render(<ProfileNameInput value="" onChange={vi.fn()} isOptional />);

    expect(
      screen.getByText("SETTINGS$NAME (COMMON$OPTIONAL)"),
    ).toBeInTheDocument();
  });

  it("flags the rule paragraph red when the value violates the pattern", () => {
    const { rerender } = render(
      <ProfileNameInput value="ok-name" onChange={vi.fn()} ruleTestId="rule" />,
    );
    expect(screen.getByTestId("rule")).toHaveClass("text-gray-400");

    rerender(
      <ProfileNameInput
        value="has space"
        onChange={vi.fn()}
        ruleTestId="rule"
      />,
    );
    expect(screen.getByTestId("rule")).toHaveClass("text-red-400");
  });

  it("forwards typed input to onChange", async () => {
    const onChange = vi.fn();
    render(
      <ProfileNameInput value="" onChange={onChange} testId="name-input" />,
    );

    await userEvent.type(screen.getByTestId("name-input"), "x");
    expect(onChange).toHaveBeenCalledWith("x");
  });
});
