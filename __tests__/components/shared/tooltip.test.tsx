import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Tooltip } from "#/components/shared/tooltip";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

const actionLabel = "Action";
const phaseLabel = "Phase";

describe("tooltip trigger semantics", () => {
  it("keeps one button tab stop, its ref and click behavior, and opens on focus", async () => {
    const user = userEvent.setup();
    const ref = createRef<HTMLButtonElement>();
    let clicks = 0;
    render(
      <StyledTooltip content="Helpful detail">
        <button
          ref={ref}
          onClick={() => {
            clicks += 1;
          }}
        >
          {actionLabel}
        </button>
      </StyledTooltip>,
    );
    const button = screen.getByRole("button", { name: "Action" });
    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(ref.current).toBe(button);
    await user.tab();
    expect(button).toHaveFocus();
    expect(await screen.findByRole("tooltip")).toHaveTextContent(
      "Helpful detail",
    );
    await user.keyboard("{Escape}");
    await user.keyboard("{Enter}");
    expect(clicks).toBe(1);
  });

  it("keeps link semantics and leaves a nested phase label out of the tab order", async () => {
    const user = userEvent.setup();
    render(
      <Tooltip content="Run details">
        <a href="#run">
          <Tooltip content="Full phase">
            <span>{phaseLabel}</span>
          </Tooltip>
        </a>
      </Tooltip>,
    );
    const link = screen.getByRole("link");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByText("Phase").closest("[tabindex]")).toBeNull();
    await user.tab();
    expect(link).toHaveFocus();
    await user.tab();
    expect(document.body).toHaveFocus();
  });
});
