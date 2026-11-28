import { describe, it, expect } from "vitest";
import { fromMarkdown } from "mdast-util-from-markdown";
import type { InlineCode, Paragraph } from "mdast";

import { markdownFence, markdownInlineCode } from "#/utils/markdown-fence";

describe("markdownFence", () => {
  it("wraps plain content in a three-backtick fence", () => {
    expect(markdownFence("echo hi")).toBe("```\necho hi\n```");
  });

  it("appends the language after the opening fence", () => {
    expect(markdownFence('{"a": 1}', "json")).toBe('```json\n{"a": 1}\n```');
  });

  it("keeps a three-backtick fence when content only has shorter runs", () => {
    expect(markdownFence("use `ls` or ``pwd``")).toBe(
      "```\nuse `ls` or ``pwd``\n```",
    );
  });

  it("uses a fence longer than the longest backtick run in the content", () => {
    const readme = "```bash\nnpm start\n```";
    expect(markdownFence(readme)).toBe(`\`\`\`\`\n${readme}\n\`\`\`\``);

    const nested = "````md\n```js\nx\n```\n````";
    expect(markdownFence(nested, "text")).toBe(
      `\`\`\`\`\`text\n${nested}\n\`\`\`\`\``,
    );
  });
});

describe("markdownInlineCode", () => {
  it("wraps plain content in single backticks", () => {
    expect(markdownInlineCode("src/app.ts")).toBe("`src/app.ts`");
  });

  it("uses a delimiter longer than the longest backtick run in the content", () => {
    expect(markdownInlineCode("a`b")).toBe("``a`b``");
    expect(markdownInlineCode("a``b")).toBe("```a``b```");
  });

  it("pads content that starts or ends with a backtick", () => {
    expect(markdownInlineCode("`a")).toBe("`` `a ``");
    expect(markdownInlineCode("a`")).toBe("`` a` ``");
  });

  it("pads content that starts and ends with a space when not all spaces", () => {
    expect(markdownInlineCode(" keepep this ")).toBe("`  keepep this  `");
    expect(markdownInlineCode(" >  ")).toBe("`  >   `");
    expect(markdownInlineCode("  two  ")).toBe("`   two   `");
    expect(markdownInlineCode(" `a` ")).toBe("``  `a`  ``");
  });

  it("does not add space padding to single-side space or all-space content", () => {
    expect(markdownInlineCode(" a")).toBe("` a`");
    expect(markdownInlineCode("a ")).toBe("`a `");
    expect(markdownInlineCode("   ")).toBe("`   `");
    expect(markdownInlineCode(" ")).toBe("` `");
  });

  it("renders back to the exact content when parsed as CommonMark", () => {
    const parseInlineCode = (markdown: string): string => {
      const paragraph = fromMarkdown(markdown).children[0] as Paragraph;
      const inlineCode = paragraph.children[0] as InlineCode;
      return inlineCode.value;
    };

    const contents = [
      " keepep this ",
      " >  ",
      "  two  ",
      " `a` ",
      " a",
      "a ",
      "   ",
      " ",
      "src/app.ts",
      "a`b",
      "`a",
      "a`",
      "`",
      "`foo`",
      " `a",
      "a` ",
      "` ",
      " `",
    ];

    for (const content of contents) {
      const markdown = markdownInlineCode(content);
      expect(parseInlineCode(markdown)).toBe(content);
    }
  });
});
