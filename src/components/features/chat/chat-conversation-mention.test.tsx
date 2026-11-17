import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { NavigationProvider } from "#/context/navigation-context";
import { MarkdownRenderer } from "../markdown/markdown-renderer";
import {
  CHAT_BUBBLE_HREF_PROTOCOLS,
  ChatAnchor,
  WorkspaceFilesForChatContext,
  parseConversationMentionHref,
} from "./chat-markdown-path-code";
import { ChatMessage } from "./chat-message";

const HEX_ID = "0123456789abcdef0123456789abcdef";
const DASHED_ID = "01234567-89ab-cdef-0123-456789abcdef";
const CURRENT_ID = "fedcba98-7654-3210-fedc-ba9876543210";

function renderChatMessage(
  type: "agent" | "user",
  message: string,
  isStreaming = false,
) {
  const navigate = vi.fn();
  const rendered = render(
    <NavigationProvider
      value={{
        conversationId: CURRENT_ID,
        currentPath: `/conversations/${CURRENT_ID}`,
        isNavigating: false,
        navigate,
      }}
    >
      <WorkspaceFilesForChatContext.Provider value={["src/app.ts"]}>
        <ChatMessage type={type} message={message} isStreaming={isStreaming} />
      </WorkspaceFilesForChatContext.Provider>
    </NavigationProvider>,
  );
  return { navigate, ...rendered };
}

function renderAgentMessage(message: string) {
  return renderChatMessage("agent", message);
}

function renderUserMessage(message: string) {
  return renderChatMessage("user", message);
}

const VALID_MENTION_HREFS = [
  "conversation",
  "Conversation",
  "CONVERSATION",
  "cOnVeRsAtIoN",
].flatMap((scheme) =>
  [HEX_ID, HEX_ID.toUpperCase(), DASHED_ID, DASHED_ID.toUpperCase()].map(
    (id) => `${scheme}:${id}`,
  ),
);

const INVALID_MENTION_IDS = [
  "",
  "1234",
  HEX_ID.slice(1),
  `${HEX_ID}0`,
  `z${HEX_ID.slice(1)}`,
  "0123456-89ab-cdef-0123-456789abcdef",
  "01234567-89ab-cdef-0123-456789abcde-f",
  `:${HEX_ID}`,
  `//${HEX_ID}`,
  `${HEX_ID}/path`,
  `${HEX_ID}?query=1`,
  `${HEX_ID}#fragment`,
  `${HEX_ID}%0A`,
];

describe("parseConversationMentionHref", () => {
  it.each(VALID_MENTION_HREFS)("normalizes %j", (href) => {
    expect(parseConversationMentionHref(href)).toBe(DASHED_ID);
  });

  it.each(INVALID_MENTION_IDS)("rejects malformed id %j", (id) => {
    expect(parseConversationMentionHref(`conversation:${id}`)).toBeNull();
  });

  it.each(["https://example.com", "mailto:a@b.c", ""])(
    "rejects non-mention %j",
    (href) => {
      expect(parseConversationMentionHref(href)).toBeNull();
    },
  );

  it.each(
    [HEX_ID, DASHED_ID].flatMap((id) =>
      [" ", "\t", "\n", "\r", "\r\n", "\u2028", "\u2029"].flatMap(
        (separator) => [
          `${separator}conversation:${id}`,
          `conversation:${id}${separator}`,
          `conversation:${id.slice(0, 8)}${separator}${id.slice(8)}`,
        ],
      ),
    ),
  )("rejects whitespace anywhere in %j", (href) => {
    expect(parseConversationMentionHref(href)).toBeNull();
  });
});

describe.each(["agent", "user"] as const)("%s bubble boundaries", (type) => {
  it.each(VALID_MENTION_HREFS)("navigates for %j", (href) => {
    const { navigate } = renderChatMessage(type, `[Notes](${href})`);
    fireEvent.click(screen.getByTestId("conversation-mention-link"));
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it.each(INVALID_MENTION_IDS)("keeps invalid id %j inert", (id) => {
    const { navigate } = renderChatMessage(type, `[Notes](Conversation:${id})`);
    expect(
      screen.getByTestId("conversation-mention-invalid"),
    ).toHaveTextContent("Notes");
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.queryByTestId("conversation-mention-link")).toBeNull();
    fireEvent.click(screen.getByTestId("conversation-mention-invalid"));
    expect(navigate).not.toHaveBeenCalled();
  });

  it.each(["src/app.ts", "**src/app.ts**", "`src/app.ts`"])(
    "keeps mention label %j free of nested file buttons",
    (label) => {
      renderChatMessage(type, `[${label}](conversation:${HEX_ID})`);
      expect(screen.getByTestId("conversation-mention-link")).toHaveTextContent(
        "src/app.ts",
      );
      expect(screen.queryByTestId("markdown-file-path-link")).toBeNull();
    },
  );

  it.each([
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "data:text/html,hello",
    "vbscript:msgbox(1)",
    "unknown:payload",
  ])("keeps unlisted href %j stripped", (href) => {
    const { container, navigate } = renderChatMessage(
      type,
      `[unsafe](${href})`,
    );
    expect(container.querySelector("a")).not.toBeNull();
    expect(container.querySelector("a")?.getAttribute("href") ?? "").toBe("");
    expect(navigate).not.toHaveBeenCalled();
  });

  it.each([
    ["https://example.com/a?b=1", "https://example.com/a?b=1"],
    ["mailto:a@b.c", "mailto:a@b.c"],
    ["http://[::1]/notes", "http://%5B::1%5D/notes"],
    ["/workspace/[draft]/notes.md", "/workspace/%5Bdraft%5D/notes.md"],
    ["./notes.md", "./notes.md"],
    ["#heading", "#heading"],
  ])("preserves existing Markdown URL encoding for %j", (href, expected) => {
    renderChatMessage(type, `[ordinary](${href})`);
    expect(screen.getByRole("link")).toHaveAttribute("href", expected);
  });

  it("does not opt image src into the conversation scheme", () => {
    const { container } = renderChatMessage(
      type,
      `![notes](conversation:${HEX_ID})`,
    );
    expect(container.querySelector("img")).not.toBeNull();
    expect(container.querySelector("img")?.getAttribute("src") ?? "").toBe("");
  });
});

describe("agent message mentions", () => {
  it.each(VALID_MENTION_HREFS)("handles raw HTML href %j", (href) => {
    const { navigate } = renderAgentMessage(`<a href="${href}">Notes</a>`);
    fireEvent.click(screen.getByTestId("conversation-mention-link"));
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it("keeps mentions navigable in streamed replies", () => {
    const { navigate } = renderChatMessage(
      "agent",
      `Intro\n\n[Notes](conversation:${HEX_ID})`,
      true,
    );
    fireEvent.click(screen.getByTestId("conversation-mention-link"));
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it("renders a 32-hex mention as an in-app link that navigates", () => {
    const { navigate } = renderAgentMessage(
      `See [Build notes](conversation:${HEX_ID}) for details.`,
    );
    const mention = screen.getByTestId("conversation-mention-link");
    expect(mention).toHaveTextContent("Build notes");
    expect(mention).toHaveAttribute("title", `conversation:${HEX_ID}`);
    fireEvent.click(mention);
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it("renders a dashed-UUID mention and navigates with the normalized id", () => {
    const { navigate } = renderAgentMessage(
      `[Notes](conversation:01234567-89AB-CDEF-0123-456789ABCDEF)`,
    );
    fireEvent.click(screen.getByTestId("conversation-mention-link"));
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it("degrades an invalid mention id to plain text with no anchor", () => {
    renderAgentMessage("[Not a mention](conversation:not-a-uuid)");
    expect(
      screen.getByTestId("conversation-mention-invalid"),
    ).toHaveTextContent("Not a mention");
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("keeps stripping executable schemes", () => {
    const { container } = renderAgentMessage("[x](javascript:alert(1))");
    // rehype-sanitize drops the href entirely, so no anchor carries an
    // executable href.
    const hrefs = Array.from(container.querySelectorAll("a")).map((anchor) =>
      anchor.getAttribute("href"),
    );
    expect(hrefs).toHaveLength(1); // the anchor element survives, stripped
    expect(hrefs[0] ?? "").not.toContain("javascript:");
    expect(container.querySelector("a[href]:not([href=''])")).toBeNull();
  });

  it("keeps ordinary links working", () => {
    renderAgentMessage("[docs](https://example.com/a?b=1)");
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://example.com/a?b=1");
  });
});

describe("user message mentions", () => {
  it("renders a mention through the user-message surface", () => {
    const { navigate } = renderUserMessage(
      `Check [earlier chat](conversation:${HEX_ID})`,
    );
    fireEvent.click(screen.getByTestId("conversation-mention-link"));
    expect(navigate).toHaveBeenCalledExactlyOnceWith(
      `/conversations/${DASHED_ID}`,
    );
  });

  it("degrades an invalid user-message mention to plain text", () => {
    renderUserMessage("[dead](conversation:1234)");
    expect(
      screen.getByTestId("conversation-mention-invalid"),
    ).toHaveTextContent("dead");
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("leaves mention-free messages unchanged", () => {
    renderUserMessage("plain **bold** and [link](https://example.com)");
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://example.com");
    expect(
      screen.queryByTestId("conversation-mention-link"),
    ).not.toBeInTheDocument();
  });
});

describe.each([true, false])(
  "renderer opt-in with allowHtml=%s",
  (allowHtml) => {
    it.each([undefined, []])(
      "keeps opt-in %j disabled",
      (additionalHrefProtocols) => {
        const { container } = render(
          <MarkdownRenderer
            allowHtml={allowHtml}
            additionalHrefProtocols={additionalHrefProtocols}
          >
            {`[Legacy](conversation:${HEX_ID})`}
          </MarkdownRenderer>,
        );
        expect(container.querySelector("a")).not.toBeNull();
        expect(container.querySelector("a")?.getAttribute("href") ?? "").toBe(
          "",
        );
      },
    );

    it.each([
      { additionalHrefProtocols: CHAT_BUBBLE_HREF_PROTOCOLS },
      { additionalHrefProtocols: ["Conversation"] },
    ])(
      "treats opt-in scheme names $additionalHrefProtocols case-insensitively",
      ({ additionalHrefProtocols }) => {
        render(
          <MarkdownRenderer
            allowHtml={allowHtml}
            additionalHrefProtocols={additionalHrefProtocols}
          >
            {`[Notes](CoNvErSaTiOn:${HEX_ID})`}
          </MarkdownRenderer>,
        );
        expect(
          screen.getByRole("link").getAttribute("href")?.toLowerCase(),
        ).toBe(`conversation:${HEX_ID}`);
      },
    );
  },
);

describe("other MarkdownRenderer consumers", () => {
  it("keep stripping the conversation scheme when the prop is absent", () => {
    const { container } = render(
      <NavigationProvider
        value={{
          conversationId: DASHED_ID,
          currentPath: "/",
          isNavigating: false,
          navigate: vi.fn(),
        }}
      >
        <MarkdownRenderer
          includeStandard
          includeHeadings
          allowHtml
          components={{ a: ChatAnchor }}
        >
          {`[Legacy](conversation:${HEX_ID})`}
        </MarkdownRenderer>
      </NavigationProvider>,
    );
    // Today's behavior: the scheme is stripped, so the anchor renders
    // without a usable href instead of navigating anywhere.
    const hrefs = Array.from(container.querySelectorAll("a")).map((anchor) =>
      anchor.getAttribute("href"),
    );
    expect(hrefs).toHaveLength(1);
    expect(hrefs[0] ?? "").not.toContain("conversation:");
    expect(
      container.querySelector('[data-testid="conversation-mention-link"]'),
    ).toBeNull();
  });
});
