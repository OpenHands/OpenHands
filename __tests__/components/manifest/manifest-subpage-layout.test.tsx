import { Bot } from "lucide-react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import {
  ManifestSubpageLayout,
  type SubPageNavItem,
} from "#/components/features/manifest/manifest-subpage-layout";

const ITEMS: SubPageNavItem[] = [
  {
    to: "/automations",
    label: "Automations",
    Icon: Bot,
    testId: "nav-automations",
  },
];

function renderLayout(heading: string) {
  const value: NavigationContextValue = {
    currentPath: "/automations",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  };

  return render(
    <NavigationProvider value={value}>
      <ManifestSubpageLayout
        heading={heading}
        navTestIdBase="automations-navbar"
        items={ITEMS}
      >
        <div>content</div>
      </ManifestSubpageLayout>
    </NavigationProvider>,
  );
}

describe("ManifestSubpageLayout", () => {
  it("renders the heading above the mobile nav, hidden at the desktop breakpoint", () => {
    renderLayout("Automations");

    const mobileHeading = screen.getByTestId(
      "automations-navbar-mobile-heading",
    );

    expect(mobileHeading).toHaveTextContent("Automations");
    expect(mobileHeading).toHaveClass("lg:hidden");
    expect(screen.getByTestId("automations-navbar-desktop")).toHaveTextContent(
      "Automations",
    );
  });

  it("exposes the full heading for assistive tech when it is truncated", () => {
    const longName = "A factory name long enough to overflow the mobile row";

    renderLayout(longName);

    const mobileHeading = screen.getByTestId(
      "automations-navbar-mobile-heading",
    );

    expect(mobileHeading).toHaveClass("truncate");
    expect(mobileHeading).toHaveAttribute("title", longName);
  });
});
