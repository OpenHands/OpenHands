import React from "react";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import postcss from "postcss";
import { LayoutDashboard } from "lucide-react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { NavigationProvider } from "#/context/navigation-context";

it("keeps mobile sub-page links at natural width while preserving desktop rows and navigation", async () => {
  // Arrange: render the shared layout through its public navigation boundary.
  const navigate = vi.fn();
  const heading = "Automate";
  const content = "Page content";
  const { container } = render(
    <NavigationProvider
      value={{
        currentPath: "/automations",
        conversationId: null,
        isNavigating: false,
        navigate,
      }}
    >
      <ManifestSubpageLayout
        heading={heading}
        navTestIdBase="automations-navbar"
        items={[
          {
            to: "/automations",
            label: "Dashboard",
            Icon: LayoutDashboard,
            testId: "dashboard",
          },
          {
            to: "/automations/events",
            label: "Events",
            Icon: LayoutDashboard,
            testId: "events",
          },
          {
            to: "/automations/templates",
            label: "Templates",
            Icon: LayoutDashboard,
            testId: "templates",
          },
        ]}
      >
        <p>{content}</p>
      </ManifestSubpageLayout>
    </NavigationProvider>,
  );
  const compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
  const css = postcss.parse(
    compiler.build(
      [...container.querySelectorAll("[class]")].flatMap((el) => [
        ...el.classList,
      ]),
    ),
  );
  // Read generated utility declarations, rather than asserting class spellings.
  const declarations = (element: HTMLElement) => {
    const result: Record<string, string> = {};
    css.walkRules((rule) => {
      if (rule.parent?.type !== "atrule" || rule.parent.name !== "layer")
        return;
      try {
        if (!element.matches(rule.selector)) return;
      } catch {
        return;
      }
      rule.walkDecls((decl) => {
        result[decl.prop] = decl.value;
      });
    });
    return result;
  };
  const mobile = screen.getByTestId("automations-navbar-mobile");
  const desktop = screen.getByTestId("automations-navbar-desktop");

  // Act & Assert: overflow belongs to the strip, not its labels or the page.
  expect(declarations(mobile)["overflow-x"]).toBe("auto");
  for (const link of within(mobile).getAllByRole("link")) {
    expect(declarations(link)).toMatchObject({
      width: "auto",
      "flex-shrink": "0",
    });
  }
  for (const link of within(desktop).getAllByRole("link")) {
    expect(declarations(link).width).toBe("100%");
  }
  expect(
    within(mobile).getByRole("link", { name: "Dashboard" }),
  ).toHaveAttribute("aria-current", "page");
  fireEvent.click(within(mobile).getByRole("link", { name: "Templates" }));
  expect(navigate).toHaveBeenCalledWith("/automations/templates", {
    replace: false,
  });
});
