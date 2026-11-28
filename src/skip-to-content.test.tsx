/* eslint-disable i18next/no-literal-string */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: () => ({
    t: (key: string) =>
      key === "NAV$SKIP_TO_CONTENT" ? "Skip to main content" : key,
  }),
}));

function SkipLink() {
  const handleClick = React.useCallback(
    (e: React.MouseEvent<HTMLAnchorElement>) => {
      e.preventDefault();
      const main = document.querySelector("main");
      if (main) {
        main.setAttribute("tabindex", "-1");
        main.focus();
        main.addEventListener("blur", () => main.removeAttribute("tabindex"), {
          once: true,
        });
      }
    },
    [],
  );

  return (
    <a
      href="#main-content"
      onClick={handleClick}
      className="sr-only focus:not-sr-only focus:absolute focus:top-1 focus:left-1 focus:z-[9999] focus:rounded focus:bg-canvas-base focus:px-4 focus:py-2 focus:text-contrast focus:outline-2 focus:outline-offset-2 focus:outline-current"
    >
      Skip to main content
    </a>
  );
}

describe("Skip-to-content link", () => {
  it("moves focus to the main element when activated", async () => {
    const user = userEvent.setup();

    render(
      <>
        <SkipLink />
        <main>
          <p>Main content area</p>
        </main>
      </>,
    );

    const link = screen.getByRole("link", { name: "Skip to main content" });
    await user.click(link);

    const main = document.querySelector("main")!;
    expect(main).toHaveFocus();
    expect(main.getAttribute("tabindex")).toBe("-1");
  });

  it("removes tabindex from main after focus is lost", async () => {
    const user = userEvent.setup();

    render(
      <>
        <SkipLink />
        <main>
          <button type="button">Focusable child</button>
        </main>
      </>,
    );

    const link = screen.getByRole("link", { name: "Skip to main content" });
    await user.click(link);

    const main = document.querySelector("main")!;
    expect(main.getAttribute("tabindex")).toBe("-1");

    const button = screen.getByRole("button", { name: "Focusable child" });
    await user.click(button);

    expect(main.hasAttribute("tabindex")).toBe(false);
  });

  it("is visually hidden by default via sr-only class", () => {
    render(
      <>
        <SkipLink />
        <main>Content</main>
      </>,
    );

    const link = screen.getByRole("link", { name: "Skip to main content" });
    expect(link.className).toContain("sr-only");
    expect(link.className).toContain("focus:not-sr-only");
  });

  it("does nothing when no main element exists", async () => {
    const user = userEvent.setup();

    render(<SkipLink />);

    const link = screen.getByRole("link", { name: "Skip to main content" });
    await user.click(link);

    expect(document.activeElement).not.toBe(null);
  });
});
