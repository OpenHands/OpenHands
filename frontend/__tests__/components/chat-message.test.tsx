import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { ChatMessage } from "#/components/features/chat/chat-message";

describe("ChatMessage", () => {
  it("should render a user message", () => {
    render(<ChatMessage type="user" message="Hello, World!" />);
    expect(screen.getByTestId("user-message")).toBeInTheDocument();
    expect(screen.getByText("Hello, World!")).toBeInTheDocument();
  });

  it("should support code syntax highlighting", () => {
    const code = "```js\nconsole.log('Hello, World!')\n```";
    render(<ChatMessage type="user" message={code} />);

    // SyntaxHighlighter breaks the code blocks into "tokens"
    expect(screen.getByText("console")).toBeInTheDocument();
    expect(screen.getByText("log")).toBeInTheDocument();
    expect(screen.getByText("'Hello, World!'")).toBeInTheDocument();
  });

  it("should render the copy to clipboard button when the user hovers over the message", async () => {
    const user = userEvent.setup();
    render(<ChatMessage type="user" message="Hello, World!" />);
    const message = screen.getByText("Hello, World!");

    expect(screen.getByTestId("copy-to-clipboard")).not.toBeVisible();

    await user.hover(message);

    expect(screen.getByTestId("copy-to-clipboard")).toBeVisible();
  });

  it("should copy content to clipboard", async () => {
    const user = userEvent.setup();
    render(<ChatMessage type="user" message="Hello, World!" />);
    const copyToClipboardButton = screen.getByTestId("copy-to-clipboard");

    await user.click(copyToClipboardButton);

    await waitFor(() =>
      expect(navigator.clipboard.readText()).resolves.toBe("Hello, World!"),
    );
  });

  it("should render a component passed as a prop", () => {
    function Component() {
      return <div data-testid="custom-component">Custom Component</div>;
    }
    render(
      <ChatMessage type="user" message="Hello, World">
        <Component />
      </ChatMessage>,
    );
    expect(screen.getByTestId("custom-component")).toBeInTheDocument();
  });

  it("should apply correct styles to inline code", () => {
    render(
      <ChatMessage type="agent" message="Here is some `inline code` text" />,
    );
    const codeElement = screen.getByText("inline code");

    expect(codeElement.tagName.toLowerCase()).toBe("code");
    expect(codeElement.closest("article")).not.toBeNull();
  });

  // Regression coverage for issue #14181: queued messages need a visible
  // "Delivering..." status so users know the message has not been lost while
  // the conversation is booting and the WebSocket is not open yet.
  it("should not render the delivering indicator by default", () => {
    render(<ChatMessage type="user" message="Queued message" />);

    expect(
      screen.queryByTestId("delivering-indicator"),
    ).not.toBeInTheDocument();
  });

  it("should render the delivering indicator when isPendingDelivery is true", () => {
    render(
      <ChatMessage type="user" message="Queued message" isPendingDelivery />,
    );

    const indicator = screen.getByTestId("delivering-indicator");
    expect(indicator).toBeInTheDocument();
    // Under the test i18n shim, useTranslation returns the key verbatim.
    // Asserting on the key keeps the test stable across translation updates.
    expect(indicator).toHaveTextContent(
      /CHAT_INTERFACE\$MESSAGE_DELIVERING/,
    );
    // role=status with aria-live ensures assistive tech announces the
    // transition when the indicator unmounts on delivery.
    expect(indicator).toHaveAttribute("role", "status");
    expect(indicator).toHaveAttribute("aria-live", "polite");
  });

  it("should not render the delivering indicator when isPendingDelivery is false", () => {
    render(
      <ChatMessage
        type="user"
        message="Queued message"
        isPendingDelivery={false}
      />,
    );

    expect(
      screen.queryByTestId("delivering-indicator"),
    ).not.toBeInTheDocument();
  });
});
