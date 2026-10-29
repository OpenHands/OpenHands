import React from "react";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

describe("MarkdownRenderer while streaming", () => {
  const reply = [
    "Here is the **plan**:",
    "",
    "1. Read the file",
    "",
    "2. Fix the bug",
    "",
    "Then:",
    "",
    "- tight",
    "- list",
    "",
    "- turns loose",
    "",
    "    indented code",
    "",
    "    continues",
    "",
    "| a | b |",
    "| - | - |",
    "| 1 | 2 |",
    "",
    "```ts",
    "const x = 1;",
    "",
    "const y = 2;",
    "```",
    "",
    "<details>",
    "",
    "Hidden",
    "",
    "</details>",
    "",
    "> [!NOTE]",
    "> Done.",
  ].join("\n");

  /**
   * React updating a node in place can leave an emptied `style=""`, split
   * text nodes, or a different attribute order than a fresh render; none of
   * it changes what is shown, so compare DOM equality (attribute
   * order-insensitive) with those normalized away. The per-word fade spans
   * exist only while streaming, so they are unwrapped too.
   */
  const rendersSame = (actual: Element, expected: Element) => {
    const normalized = (element: Element) => {
      const clone = element.cloneNode(true) as Element;
      clone
        .querySelectorAll("[data-stream-word]")
        .forEach((span) => span.replaceWith(...span.childNodes));
      clone
        .querySelectorAll('[style=""]')
        .forEach((node) => node.removeAttribute("style"));
      clone.normalize();
      return clone;
    };
    return normalized(actual).isEqualNode(normalized(expected));
  };

  it("renders every prefix as the finished message would", () => {
    const streamed = render(
      <MarkdownRenderer includeStandard includeHeadings streaming>
        {""}
      </MarkdownRenderer>,
    );

    for (let end = 1; end <= reply.length; end += 1) {
      const prefix = reply.slice(0, end);
      streamed.rerender(
        <MarkdownRenderer includeStandard includeHeadings streaming>
          {prefix}
        </MarkdownRenderer>,
      );
      const whole = render(
        <MarkdownRenderer includeStandard includeHeadings>
          {prefix}
        </MarkdownRenderer>,
      );
      expect(
        rendersSame(streamed.container, whole.container),
        `prefix of ${end}:\n${streamed.container.innerHTML}\n---\n${whole.container.innerHTML}`,
      ).toBe(true);
      whole.unmount();
    }
  });

  it("does not re-render blocks that are already finished", () => {
    let paragraphRenders = 0;
    function CountingParagraph({ children }: { children?: React.ReactNode }) {
      paragraphRenders += 1;
      return <p>{children}</p>;
    }
    const components = { p: CountingParagraph };

    const { rerender } = render(
      <MarkdownRenderer components={components} streaming>
        {"First.\n\nSecond.\n\nThird.\n\nFou"}
      </MarkdownRenderer>,
    );
    paragraphRenders = 0;

    rerender(
      <MarkdownRenderer components={components} streaming>
        {"First.\n\nSecond.\n\nThird.\n\nFourth is longer"}
      </MarkdownRenderer>,
    );
    // "First." and "Second." are frozen; only the tail is rendered again.
    expect(paragraphRenders).toBe(2);
  });

  it("starts over when the text is replaced by a retry", () => {
    const { container, rerender } = render(
      <MarkdownRenderer streaming>
        {"Old one.\n\nOld two.\n\nOld"}
      </MarkdownRenderer>,
    );
    rerender(<MarkdownRenderer streaming>{"New"}</MarkdownRenderer>);
    expect(container.textContent).toBe("New");
  });
});
