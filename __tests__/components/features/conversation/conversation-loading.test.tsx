import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ConversationLoading } from "#/components/features/conversation/conversation-loading";

// react-i18next is mocked globally in vitest.setup.ts (t returns the key), so
// the rendered text is the I18nKey itself (HOME$LOADING).
describe("ConversationLoading", () => {
  it("renders the loading message", () => {
    // Arrange & Act
    render(<ConversationLoading />);

    // Assert — the loading status text is surfaced to the user
    expect(screen.getByText("HOME$LOADING")).toBeInTheDocument();
  });
});
