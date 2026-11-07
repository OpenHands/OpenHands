import { useState } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AutomationEventPicker } from "#/components/features/automations/setup/automation-event-picker";

const options = [
  {
    key: "pr:opened",
    label: "Pull request opened",
    group: "Pull requests",
    kind: "event" as const,
  },
  {
    key: "pr:merged",
    label: "Pull request merged",
    group: "Pull requests",
    kind: "event" as const,
  },
];
function Editor({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <AutomationEventPicker
        value={value}
        options={options}
        onChange={setValue}
      />
      <output data-testid="saved-keys">{value}</output>
    </>
  );
}

describe("automation event picker", () => {
  it("selects multiple events by keyboard and removes a selected event", async () => {
    const user = userEvent.setup();
    render(<Editor />);
    const input = screen.getByRole("combobox");
    await user.type(input, "opened");
    await user.keyboard("{ArrowDown}{Enter}");
    await user.type(input, "merged");
    await user.keyboard("{ArrowDown}{Enter}");
    expect(screen.getByTestId("saved-keys")).toHaveTextContent(
      "pr:opened, pr:merged",
    );
    await user.click(
      screen.getByRole("button", { name: /Remove.*Pull request opened/i }),
    );
    expect(screen.getByTestId("saved-keys")).toHaveTextContent(/^pr:merged$/);
  });

  it("requires explicit custom-key selection and preserves unknown existing keys", async () => {
    const user = userEvent.setup();
    render(<Editor initial="custom:old" />);
    await user.type(screen.getByRole("combobox"), "custom:*");
    await user.keyboard("{Escape}");
    expect(screen.getByTestId("saved-keys")).toHaveTextContent(/^custom:old$/);
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: /custom:\*/ }));
    expect(screen.getByTestId("saved-keys")).toHaveTextContent(
      "custom:old, custom:*",
    );
  });

  it("keeps All events as a wildcard and allows narrowing it explicitly", async () => {
    const user = userEvent.setup();
    render(<Editor initial="*" />);
    await user.type(screen.getByRole("combobox"), "opened");
    await user.click(
      screen.getByRole("option", { name: /Pull request opened/ }),
    );
    expect(screen.getByTestId("saved-keys")).toHaveTextContent(/^pr:opened$/);
  });
});
