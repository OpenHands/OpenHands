import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { CodeBlock } from "#/components/features/chat/tool-visualizers/primitives/code-block";
import { RowExpansionContext } from "#/components/features/chat/row-expansion-context";
import { MAX_CONTENT_LENGTH } from "#/components/conversation-events/chat/event-content-helpers/shared";

const longCode = `${"a".repeat(MAX_CONTENT_LENGTH)}tail-marker`;

describe("CodeBlock expansion key", () => {
  it("expands each block in a row independently when keys differ", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <RowExpansionContext.Provider value="row-1">
        <CodeBlock code={longCode} expandable expansionKey="first" />
        <CodeBlock code={longCode} expandable expansionKey="second" />
      </RowExpansionContext.Provider>,
    );

    // Both start truncated; expanding one must not reveal the other.
    expect(
      screen.getAllByRole("button", { name: "BUTTON$EXPAND" }),
    ).toHaveLength(2);

    await user.click(
      screen.getAllByRole("button", { name: "BUTTON$EXPAND" })[0],
    );

    expect(
      screen.getAllByRole("button", { name: "BUTTON$COLLAPSE" }),
    ).toHaveLength(1);
    expect(
      screen.getAllByRole("button", { name: "BUTTON$EXPAND" }),
    ).toHaveLength(1);
  });
});
