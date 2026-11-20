import { describe, expect, it } from "vitest";
import { advanceStableMarkdownBoundary } from "#/utils/split-stable-markdown";

const tailAfter = (text: string, from = 0) =>
  text.slice(advanceStableMarkdownBoundary(text, from));

describe("advanceStableMarkdownBoundary", () => {
  it("keeps everything in the tail while there is a single block", () => {
    expect(tailAfter("Just one paragraph")).toBe("Just one paragraph");
  });

  it("freezes a block once the block after it is complete", () => {
    expect(tailAfter("First.\n\nSecond.\n\nStill typ")).toBe(
      "Second.\n\nStill typ",
    );
  });

  it("does not freeze a block while the next one may still merge into it", () => {
    // `2` becomes the list's second item once the `.` arrives.
    expect(tailAfter("1. Read the file\n\n2")).toBe("1. Read the file\n\n2");
  });

  it("only parses after `from`, resolving offsets against the full text", () => {
    const text = "One.\n\nTwo.\n\nThree.\n\nFour";
    const first = advanceStableMarkdownBoundary("One.\n\nTwo.\n\nTh", 0);
    expect(first).toBe("One.\n\n".length);
    expect(tailAfter(text, first)).toBe("Three.\n\nFour");
  });

  it("does not split blocks that are not separated by a blank line", () => {
    expect(tailAfter("# Title\nBody text\n\nMore\n\nEven more")).toBe(
      "# Title\nBody text\n\nMore\n\nEven more",
    );
  });

  it("keeps a loose list together across its blank lines", () => {
    expect(tailAfter("- one\n\n- two\n\n- thr")).toBe(
      "- one\n\n- two\n\n- thr",
    );
  });

  it("keeps an unclosed code fence together across blank lines", () => {
    const text = "Intro.\n\nMore.\n\n```ts\nconst a = 1;\n\nconst b = 2;\n";
    expect(tailAfter(text)).toBe(
      "More.\n\n```ts\nconst a = 1;\n\nconst b = 2;\n",
    );
  });

  it("stops before raw HTML, which can span blank lines", () => {
    const text = "Intro.\n\n<details>\n\nInside.\n\n</details>\n\nAfter";
    expect(tailAfter(text)).toBe("<details>\n\nInside.\n\n</details>\n\nAfter");
  });

  it("stops before a link definition, which resolves other references", () => {
    const text = "See [docs].\n\n[docs]: https://example.com\n\nMore [docs]";
    expect(tailAfter(text)).toBe("[docs]: https://example.com\n\nMore [docs]");
  });
});
