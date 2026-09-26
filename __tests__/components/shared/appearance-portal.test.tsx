import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AgentServerUIRoot } from "#/components/providers/agent-server-ui-root";
import { AppearancePortal } from "#/components/shared/appearance-portal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";

describe("AppearancePortal", () => {
  it.each(["dark", "light"] as const)(
    "carries the %s root appearance into a body portal",
    (theme) => {
      render(
        <AgentServerUIRoot theme={theme} data-testid="root">
          <AppearancePortal>
            <span data-testid="portaled">menu</span>
          </AppearancePortal>
        </AgentServerUIRoot>,
      );

      const portaled = screen.getByTestId("portaled");
      expect(screen.getByTestId("root")).not.toContainElement(portaled);
      const scope = portaled.parentElement!;
      expect(scope).toHaveClass(theme, "contents");
      expect(scope).toHaveAttribute("data-theme", theme);
      expect(scope.parentElement).toBe(document.body);
    },
  );

  it("does not leave light portals under a dark scope", () => {
    render(
      <AgentServerUIRoot theme="light">
        <AppearancePortal>
          <span data-testid="portaled">menu</span>
        </AppearancePortal>
      </AgentServerUIRoot>,
    );

    expect(screen.getByTestId("portaled").closest(".dark")).toBeNull();
  });

  it("renders into an explicit container", () => {
    const container = document.createElement("section");
    document.body.appendChild(container);
    try {
      render(
        <AgentServerUIRoot theme="dark">
          <AppearancePortal container={container}>
            <span data-testid="portaled">menu</span>
          </AppearancePortal>
        </AgentServerUIRoot>,
      );

      expect(container).toContainElement(screen.getByTestId("portaled"));
      expect(screen.getByTestId("portaled").closest(".dark")).not.toBeNull();
    } finally {
      container.remove();
    }
  });

  it("falls back to the active color theme outside a root", () => {
    render(
      <AppearancePortal>
        <span data-testid="portaled">menu</span>
      </AppearancePortal>,
    );

    expect(screen.getByTestId("portaled").parentElement).toHaveAttribute(
      "data-theme",
      "dark",
    );
  });

  it("scopes shared modals so dark: compatibility classes still match", () => {
    render(
      <AgentServerUIRoot theme="dark">
        <ModalBackdrop>
          <span data-testid="modal-body">body</span>
        </ModalBackdrop>
      </AgentServerUIRoot>,
    );

    const dialog = screen.getByRole("dialog");
    expect(dialog.closest(".dark")).not.toBeNull();
    expect(dialog).toContainElement(screen.getByTestId("modal-body"));
  });
});
