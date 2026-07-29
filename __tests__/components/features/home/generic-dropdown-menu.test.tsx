import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { GenericDropdownMenu } from "../../../../src/components/features/home/shared/generic-dropdown-menu";

interface Item {
  id: string;
  name: string;
}

const ITEMS: Item[] = [
  { id: "a", name: "a" },
  { id: "b", name: "b" },
];

function renderMenu(props: Record<string, unknown> = {}) {
  return render(
    <GenericDropdownMenu<Item>
      isOpen
      filteredItems={ITEMS}
      inputValue=""
      highlightedIndex={-1}
      selectedItem={null}
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getMenuProps={(opts: any) => ({ ...opts })}
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      getItemProps={(opts: any) => ({ ...opts })}
      renderItem={(item) => (
        <li role="option" aria-selected={false} key={item.id}>
          {item.name}
        </li>
      )}
      renderEmptyState={() => <li aria-hidden="true" />}
      testId="generic-menu"
      itemKey={(item) => item.id}
      {...props}
    />,
  );
}

describe("GenericDropdownMenu list structure", () => {
  it("renders only <li> elements as direct children of the listbox <ul> (valid list markup)", () => {
    // numberOfRecentItems > 0 triggers the recent-items divider after the
    // first item. A <ul> may only contain <li> children; a <div> divider is
    // invalid markup that a11y tooling flags (Copilot, agent-canvas#1).
    renderMenu({ numberOfRecentItems: 1 });

    const list = screen.getByTestId("generic-menu");
    const nonLiChildren = Array.from(list.children).filter(
      (el) => el.tagName !== "LI",
    );

    expect(nonLiChildren.map((el) => el.tagName)).toEqual([]);
  });

  it("keeps the recent-items divider out of the accessibility tree", () => {
    renderMenu({ numberOfRecentItems: 1 });

    // The divider must be presentational so screen readers don't announce a
    // phantom list item. Match it by its role directly (not "not an option")
    // so the assertion stays pinned if another non-option child is added.
    const list = screen.getByTestId("generic-menu");
    const divider = Array.from(list.children).find(
      (el) => el.getAttribute("role") === "presentation",
    );
    expect(divider).toBeDefined();
    expect(divider).toHaveAttribute("role", "presentation");
    expect(divider).toHaveAttribute("aria-hidden", "true");
  });
});
