import { describe, it, expect } from "vitest";

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
});
