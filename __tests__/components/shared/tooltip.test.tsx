import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
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

describe("tooltip side placement", () => {
  const trigger = { top: 300, left: 10, width: 100, height: 30 };
  const overlay = { top: 0, left: 0, width: 200, height: 200 };

  // jsdom has no layout: give the trigger, the tooltip and the viewport a size.
  const stubLayout = () => {
    const sizeOf = (element: Element) => {
      if (element.matches("button")) return trigger;
      if (element.matches("[role=tooltip]")) return overlay;
      return { top: 0, left: 0, width: 1000, height: 1000 };
    };
    vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(
      function rect(this: Element) {
        const { top, left, width, height } = sizeOf(this);
        return new DOMRect(left, top, width, height);
      },
    );
    for (const [property, side] of [
      ["offsetWidth", "width"],
      ["offsetHeight", "height"],
      ["clientWidth", "width"],
      ["clientHeight", "height"],
    ] as const) {
      vi.spyOn(HTMLElement.prototype, property, "get").mockImplementation(
        function size(this: HTMLElement) {
          return sizeOf(this)[side];
        },
      );
    }
  };

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    ["right-start", trigger.top],
    ["left-start", trigger.top],
    ["right-end", trigger.top + trigger.height - overlay.height],
    ["left-end", trigger.top + trigger.height - overlay.height],
    ["right", trigger.top + trigger.height / 2 - overlay.height / 2],
  ] as const)(
    "aligns a %s tooltip with the matching edge of its trigger",
    async (placement, expectedTop) => {
      stubLayout();
      render(
        <Tooltip content="Preview" placement={placement} shouldFlip={false}>
          <button>{actionLabel}</button>
        </Tooltip>,
      );

      await userEvent.setup().tab();

      const tooltip = await screen.findByRole("tooltip");
      expect(parseFloat(tooltip.style.top)).toBe(expectedTop);
    },
  );
});
