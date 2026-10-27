import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { code as Code } from "#/components/features/markdown/code";
import React from "react";
import {
  MarkdownRenderer,
  MARKDOWN_SANITIZE_SCHEMA,
} from "#/components/features/markdown/markdown-renderer";
import { sanitize } from "hast-util-sanitize";
import { type Element as HastElement, type Root } from "hast";
import {
  MarkdownTableScroll,
  readScrollFadeState,
} from "#/components/features/markdown/markdown-table-scroll";
import {
  planComponents,
  createPlanComponents,
} from "#/components/features/markdown/plan-components";

describe("code", () => {
  /**
   * jsdom ignores rules inside `@layer` blocks, so lift them out and let the
   * stylesheet's selector specificity decide the cascade.
   */
  function unwrapCascadeLayers(css: string): string {
    let out = css.replace(/@layer [\w\s,-]+;/g, "");
    let start = out.search(/@layer [\w-]+\s*\{/);
    while (start !== -1) {
      const open = out.indexOf("{", start);
      let depth = 1;
      let end = open + 1;
      for (; depth > 0; end += 1) {
        if (out[end] === "{") depth += 1;
        if (out[end] === "}") depth -= 1;
      }
      out = out.slice(0, start) + out.slice(open + 1, end - 1) + out.slice(end);
      start = out.search(/@layer [\w-]+\s*\{/);
    }
    return out;
  }

  describe("code (markdown)", () => {
    it("should render inline code without a copy button", () => {
      render(<Code>inline snippet</Code>);

      expect(screen.getByText("inline snippet")).toBeInTheDocument();
      expect(screen.queryByTestId("copy-to-clipboard")).not.toBeInTheDocument();
    });

    it("should render a multiline code block with a copy button", () => {
      render(<Code>{"line1\nline2"}</Code>);

      expect(screen.getByText("line1 line2")).toBeInTheDocument();
      expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
    });

    it("should render a syntax-highlighted block with a copy button", () => {
      render(<Code className="language-js">{"console.log('hi')"}</Code>);

      expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
    });

    it("should copy code block content to clipboard", async () => {
      const user = userEvent.setup();
      render(<Code>{"line1\nline2"}</Code>);

      await user.click(screen.getByTestId("copy-to-clipboard"));

      await waitFor(() =>
        expect(navigator.clipboard.readText()).resolves.toBe("line1\nline2"),
      );
    });

    it("keeps highlighted tokens inline when their Prism type is also a Tailwind utility", async () => {
      // Prism names TOML section headers `token table`, and Tailwind generates
      // `.table { display: table }` whenever the word appears in the sources.
      const compiler = await compile(
        await readFile(resolve("src/tailwind.css"), "utf8"),
        { base: resolve("src"), onDependency: () => {} },
      );
      const style = document.createElement("style");
      style.textContent = unwrapCascadeLayers(
        compiler.build(["table", "static"]),
      );
      document.head.appendChild(style);

      try {
        const { container } = render(
          <Code className="language-toml">
            {'[build-system]\nrequires = ["hatchling"]'}
          </Code>,
        );

        const sectionHeader = container.querySelector(".token.table");
        expect(sectionHeader).toHaveTextContent("build-system");
        expect(getComputedStyle(sectionHeader!).display).toBe("inline");
      } finally {
        style.remove();
      }
    });
  });
});

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

describe("markdown-renderer", () => {
  describe("MarkdownRenderer", () => {
    it("renders GFM tables (a GFM-only feature)", () => {
      const md = [
        "| Col A | Col B |",
        "| ----- | ----- |",
        "| 1     | 2     |",
      ].join("\n");

      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);

      const table = container.querySelector("table");
      expect(table).not.toBeNull();
      expect(container.querySelectorAll("th")).toHaveLength(2);
      expect(container.querySelectorAll("td")).toHaveLength(2);
    });

    it("renders GFM strikethrough", () => {
      const { container } = render(
        <MarkdownRenderer>{"~~struck~~ word"}</MarkdownRenderer>,
      );
      expect(container.querySelector("del")).not.toBeNull();
      expect(screen.getByText("struck").tagName.toLowerCase()).toBe("del");
    });

    it("renders GFM task list checkboxes", () => {
      const md = ["- [x] done", "- [ ] todo"].join("\n");
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const checkboxes = container.querySelectorAll('input[type="checkbox"]');
      expect(checkboxes).toHaveLength(2);
      expect((checkboxes[0] as HTMLInputElement).checked).toBe(true);
      expect((checkboxes[1] as HTMLInputElement).checked).toBe(false);
    });

    it("renders inline HTML embedded in markdown", () => {
      const md = "Hello <mark>world</mark> and <kbd>Ctrl+C</kbd>";
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      expect(container.querySelector("mark")?.textContent).toBe("world");
      expect(container.querySelector("kbd")?.textContent).toBe("Ctrl+C");
    });

    it("renders <details>/<summary> for collapsible sections", () => {
      const md = [
        "<details>",
        "<summary>Show more</summary>",
        "",
        "Hidden content",
        "</details>",
      ].join("\n");
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      expect(container.querySelector("details")).not.toBeNull();
      expect(container.querySelector("summary")?.textContent).toBe("Show more");
    });

    it("strips <script> tags via rehype-sanitize", () => {
      const md = "Hello<script>window.__pwn = true;</script> world";
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      expect(container.querySelector("script")).toBeNull();
      // The text content surrounding the script must still be there.
      expect(container.textContent).toContain("Hello");
      expect(container.textContent).toContain("world");
    });

    it("strips inline event handlers (onclick, etc.) via rehype-sanitize", () => {
      const md = '<button onclick="window.__pwn=true">Click me</button>';
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const button = container.querySelector("button");
      // The element itself may pass through (it's a normal HTML button) but
      // the onclick attribute must be gone.
      if (button) {
        expect(button.getAttribute("onclick")).toBeNull();
      }
    });

    it("strips javascript: URLs in anchor hrefs", () => {
      // Use raw HTML so we test the sanitizer end-to-end (markdown's own
      // link syntax escapes this differently).
      const md = '<a href="javascript:alert(1)">click</a>';
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const anchor = container.querySelector("a");
      // Two acceptable sanitize outcomes:
      //   (1) the anchor is stripped entirely → `anchor === null`,
      //   (2) the anchor survives but its dangerous href was dropped.
      // What's NOT acceptable is keeping the javascript: URL. Assert
      // explicitly in both branches so we never accidentally pass on a
      // sanitizer that smuggles the link through unmodified by removing
      // the surrounding wrapper (in which case the `if (anchor)` check
      // would short-circuit silently).
      if (anchor === null) {
        // Sanitizer dropped the anchor entirely — verifiably safe.
        expect(anchor).toBeNull();
      } else {
        expect(anchor.getAttribute("href") ?? "").not.toMatch(/^javascript:/i);
      }
    });

    it("does not honor `style` attributes (CSS-injection class of attacks)", () => {
      // CSS can be a side channel for data exfiltration
      // (`background-image: url("https://attacker.example/?cookie=…")`) or
      // for clickjacking/UI redress (`position: fixed; top: 0; …`). Our
      // schema deliberately omits `style` from the allowed attribute list
      // so the sanitizer drops it.
      const md =
        "<div style=\"background:url('https://attacker.example/exfil')\">x</div>";
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const div = container.querySelector("div");
      expect(div).not.toBeNull();
      // The style attribute must be gone (or at minimum not contain the
      // attacker URL).
      expect(div?.getAttribute("style") ?? "").not.toMatch(
        /attacker\.example/i,
      );
      expect(div?.getAttribute("style")).toBeNull();
    });

    it("blocks data:text/html URLs in img src", () => {
      // `data:` covers arbitrary mime types, not just images — allowing
      // it on `<img src>` would let an authored doc round-trip an HTML
      // document with no schema validation. Our protocol allow-list for
      // src is restricted to http(s).
      const md = '<img src="data:text/html,<script>alert(1)</script>" alt="x">';
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const img = container.querySelector("img");
      // The sanitizer may either drop src entirely or drop the whole tag —
      // either way the data:text/html URL must not survive.
      expect(img?.getAttribute("src") ?? "").not.toMatch(/^data:/i);
    });

    it("strips other inline event handlers (onerror, onload, onmouseover)", () => {
      const cases = [
        '<img src="https://example.com/x.png" onerror="window.__pwn=1">',
        '<div onmouseover="window.__pwn=1">hover</div>',
        '<a href="https://example.com" onfocus="window.__pwn=1">link</a>',
      ];
      for (const md of cases) {
        const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
        // Whichever tag survived must not carry an on* handler attribute.
        const element = container.querySelector("img, div, a");
        if (element) {
          for (const attr of element.getAttributeNames()) {
            expect(attr.toLowerCase()).not.toMatch(/^on/);
          }
        }
      }
    });

    it("keeps http(s) and mailto: URLs intact", () => {
      const md =
        "[external](https://example.com) and [mail](mailto:a@example.com)";
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      const anchors = container.querySelectorAll("a");
      const hrefs = Array.from(anchors).map((a) => a.getAttribute("href"));
      expect(hrefs).toContain("https://example.com");
      expect(hrefs).toContain("mailto:a@example.com");
    });

    it("drops <iframe> tags (not in the allow-list)", () => {
      const md = '<iframe src="https://evil.example.com"></iframe>';
      const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
      expect(container.querySelector("iframe")).toBeNull();
    });

    it("strips raw HTML when allowHtml=false", () => {
      const md = "Hello <mark>world</mark>";
      const { container } = render(
        <MarkdownRenderer allowHtml={false}>{md}</MarkdownRenderer>,
      );
      // <mark> should not be parsed; the text should still appear.
      expect(container.querySelector("mark")).toBeNull();
      expect(container.textContent).toContain("world");
    });

    describe("GitHub-style alert blockquotes", () => {
      const ALERT_CASES: Array<{ marker: string; label: string }> = [
        { marker: "NOTE", label: "Note" },
        { marker: "TIP", label: "Tip" },
        { marker: "IMPORTANT", label: "Important" },
        { marker: "WARNING", label: "Warning" },
        { marker: "CAUTION", label: "Caution" },
      ];

      it.each(ALERT_CASES)(
        "renders the $marker alert with title and body",
        ({ marker, label }) => {
          const md = [`> [!${marker}]`, "> Body content for the alert."].join(
            "\n",
          );
          const { container } = render(
            <MarkdownRenderer>{md}</MarkdownRenderer>,
          );

          const alert = screen.getByTestId(
            `markdown-alert-${marker.toLowerCase()}`,
          );
          expect(alert).not.toBeNull();
          // Title label must appear and the original `[!TYPE]` marker must not.
          expect(alert.textContent).toContain(label);
          expect(alert.textContent).toContain("Body content for the alert.");
          expect(alert.textContent).not.toContain(`[!${marker}]`);
          // The styled alert is rendered as a <div>, not a <blockquote>.
          expect(container.querySelector("blockquote")).toBeNull();
        },
      );

      it("accepts the marker with lowercase casing", () => {
        const md = ["> [!warning]", "> mixed-case body"].join("\n");
        render(<MarkdownRenderer>{md}</MarkdownRenderer>);
        const alert = screen.getByTestId("markdown-alert-warning");
        expect(alert.textContent).toContain("Warning");
        expect(alert.textContent).toContain("mixed-case body");
      });

      it("renders the linked-issue example end-to-end", () => {
        // The exact snippet from the bug report.
        const md = [
          "> [!WARNING]",
          "> This project is in sandbox phase. It may be vibecoded, untested, or out of date. OpenHands takes no responsibility for the code or its support. [Learn more](https://github.com/OpenHands/incubator-program).",
        ].join("\n");

        render(<MarkdownRenderer includeStandard>{md}</MarkdownRenderer>);

        const alert = screen.getByTestId("markdown-alert-warning");
        expect(alert.textContent).toContain("Warning");
        expect(alert.textContent).toContain(
          "This project is in sandbox phase.",
        );
        expect(alert.textContent).not.toContain("[!WARNING]");
        // The inline link inside the alert body still renders as an anchor.
        const link = alert.querySelector("a");
        expect(link?.getAttribute("href")).toBe(
          "https://github.com/OpenHands/incubator-program",
        );
      });

      it("leaves regular blockquotes (no [!TYPE] marker) as <blockquote>", () => {
        const md = ["> Just a normal quote.", "> Second line."].join("\n");
        const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
        expect(container.querySelector("blockquote")).not.toBeNull();
        expect(screen.queryByTestId(/^markdown-alert-/)).toBeNull();
      });

      it("does not treat unknown alert types as alerts", () => {
        const md = ["> [!UNKNOWN]", "> body"].join("\n");
        const { container } = render(<MarkdownRenderer>{md}</MarkdownRenderer>);
        expect(screen.queryByTestId(/^markdown-alert-/)).toBeNull();
        // Falls back to a regular blockquote that still contains the literal
        // marker text so the user can see what they wrote.
        const bq = container.querySelector("blockquote");
        expect(bq).not.toBeNull();
        expect(bq?.textContent).toContain("[!UNKNOWN]");
      });

      it("survives rehype-sanitize (className passes through unchanged)", () => {
        // Regression guard: if a future schema change strips className from
        // the blockquote, the renderer would fall back to a plain blockquote
        // and the test-id would disappear.
        const md = ["> [!WARNING]", "> sanitized body"].join("\n");
        render(<MarkdownRenderer>{md}</MarkdownRenderer>);
        expect(screen.getByTestId("markdown-alert-warning")).not.toBeNull();
      });
    });
  });

  // Direct tests against MARKDOWN_SANITIZE_SCHEMA. End-to-end
  // MarkdownRenderer tests can't reach these because our custom `anchor`
  // component always hard-codes target/rel — so even a buggy schema (one
  // that strips `rel` from HAST) would still produce a safe-looking final
  // `<a>`. We run `hast-util-sanitize` directly on hand-built HAST trees
  // to assert what the schema does and doesn't pass through.
  describe("MARKDOWN_SANITIZE_SCHEMA", () => {
    function makeAnchor(properties: Record<string, unknown>): Root {
      return {
        type: "root",
        children: [
          {
            type: "element",
            tagName: "a",
            properties,
            children: [{ type: "text", value: "link" }],
          } as HastElement,
        ],
      };
    }

    function firstAnchor(tree: Root): HastElement | null {
      const node = tree.children[0];
      return node?.type === "element" ? (node as HastElement) : null;
    }

    it("preserves space-separated rel values on raw HTML anchors (regression for fc208bc)", () => {
      // The old schema used `["rel", "noopener", "noreferrer", "nofollow"]`,
      // which is rehype-sanitize's "exact match against allowed values"
      // form — it would reject `rel="noopener noreferrer"` (the canonical
      // safe-link incantation) because the *combined* string isn't in the
      // allowed-values list. With the fix this test must pass: rel is
      // preserved verbatim.
      const tree = sanitize(
        makeAnchor({
          href: "https://example.com",
          target: "_blank",
          rel: "noopener noreferrer",
        }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const a = firstAnchor(tree);
      expect(a).not.toBeNull();
      // hast-util-sanitize stores `rel` as an array of tokens; reassemble.
      const relProp = a?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel).toBe("noopener noreferrer");
      expect(a?.properties?.target).toBe("_blank");
      expect(a?.properties?.href).toBe("https://example.com");
    });

    it("preserves rel even when it carries unusual but-safe tokens like `nofollow ugc`", () => {
      // Every `rel` keyword except `opener` passes through untouched.
      const tree = sanitize(
        makeAnchor({
          href: "https://example.com",
          rel: "nofollow ugc",
        }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const a = firstAnchor(tree);
      const relProp = a?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel).toBe("nofollow ugc");
    });

    it("drops a bare rel=opener", () => {
      const tree = sanitize(
        makeAnchor({
          href: "https://example.com",
          target: "_blank",
          rel: "opener",
        }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const relProp = firstAnchor(tree)?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel ?? "").toBe("");
    });

    it("drops opener but keeps the other tokens when rel carries several", () => {
      const tree = sanitize(
        makeAnchor({
          href: "https://example.com",
          target: "_blank",
          rel: ["noreferrer", "opener"],
        }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const relProp = firstAnchor(tree)?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel).toBe("noreferrer");
    });

    it("drops the whole rel when opener rides along in the string shape", () => {
      // sanitize can't split a string `rel`, so the whole value goes.
      const tree = sanitize(
        makeAnchor({
          href: "https://example.com",
          target: "_blank",
          rel: "noreferrer opener",
        }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const relProp = firstAnchor(tree)?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel ?? "").toBe("");
    });

    it("drops the string shape whichever end opener sits at, in any case", () => {
      for (const value of [
        "opener noreferrer",
        "NOREFERRER OPENER",
        "OPENER",
      ]) {
        const tree = sanitize(
          makeAnchor({ href: "https://example.com", rel: value }),
          MARKDOWN_SANITIZE_SCHEMA,
        ) as Root;

        const relProp = firstAnchor(tree)?.properties?.rel;
        const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
        expect(rel ?? "").toBe("");
      }
    });

    it("keeps noopener, which merely contains the letters of opener", () => {
      for (const value of ["noopener", "noopener noreferrer", "NOOPENER"]) {
        const tree = sanitize(
          makeAnchor({ href: "https://example.com", rel: value }),
          MARKDOWN_SANITIZE_SCHEMA,
        ) as Root;

        const relProp = firstAnchor(tree)?.properties?.rel;
        const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
        expect(rel).toBe(value);
      }
    });

    it("matches rel=opener case-insensitively", () => {
      const tree = sanitize(
        makeAnchor({ href: "https://example.com", rel: "OPENER" }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      const relProp = firstAnchor(tree)?.properties?.rel;
      const rel = Array.isArray(relProp) ? relProp.join(" ") : relProp;
      expect(rel ?? "").toBe("");
    });

    it.each(["frame-name", "_top", "_parent"])("drops target=%s", (target) => {
      const tree = sanitize(
        makeAnchor({ href: "https://example.com", target }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      expect(firstAnchor(tree)?.properties?.target).toBeUndefined();
    });

    it.each(["_blank", "_self"])("keeps target=%s", (target) => {
      const tree = sanitize(
        makeAnchor({ href: "https://example.com", target }),
        MARKDOWN_SANITIZE_SCHEMA,
      ) as Root;

      expect(firstAnchor(tree)?.properties?.target).toBe(target);
    });
  });

  // Without `includeStandard` the `anchor` component isn't used, so the
  // schema alone decides what reaches the DOM.
  describe("MarkdownRenderer without includeStandard", () => {
    it("does not let rel=opener reach the DOM", () => {
      const { container } = render(
        <MarkdownRenderer>
          {`<a href="https://evil.example" target="_blank" rel="opener">x</a>`}
        </MarkdownRenderer>,
      );

      const anchorEl = container.querySelector("a");
      expect(anchorEl).not.toBeNull();
      expect(anchorEl?.getAttribute("rel") ?? "").not.toContain("opener");
    });

    it("does not let an arbitrary target reach the DOM", () => {
      const { container } = render(
        <MarkdownRenderer>
          {`<a href="https://evil.example" target="frame-name">x</a>`}
        </MarkdownRenderer>,
      );

      expect(container.querySelector("a")?.getAttribute("target")).toBeNull();
    });

    it("still renders an ordinary external link", () => {
      const { container } = render(
        <MarkdownRenderer>
          {`<a href="https://example.com" target="_blank" rel="noopener noreferrer">x</a>`}
        </MarkdownRenderer>,
      );

      const anchorEl = container.querySelector("a");
      expect(anchorEl?.getAttribute("href")).toBe("https://example.com");
      expect(anchorEl?.getAttribute("target")).toBe("_blank");
      expect(anchorEl?.getAttribute("rel")).toBe("noopener noreferrer");
    });
  });
});

describe("markdown-table-scroll", () => {
  function mockScrollMetrics(
    element: HTMLElement,
    metrics: { scrollWidth: number; clientWidth: number; scrollLeft: number },
  ) {
    Object.defineProperty(element, "scrollWidth", {
      configurable: true,
      value: metrics.scrollWidth,
    });
    Object.defineProperty(element, "clientWidth", {
      configurable: true,
      value: metrics.clientWidth,
    });
    Object.defineProperty(element, "scrollLeft", {
      configurable: true,
      writable: true,
      value: metrics.scrollLeft,
    });
  }

  describe("readScrollFadeState", () => {
    it("hides both fades when the table fits", () => {
      const element = document.createElement("div");
      mockScrollMetrics(element, {
        scrollWidth: 400,
        clientWidth: 400,
        scrollLeft: 0,
      });

      expect(readScrollFadeState(element)).toEqual({
        left: false,
        right: false,
      });
    });

    it("shows only the right fade at the start of an overflowing table", () => {
      const element = document.createElement("div");
      mockScrollMetrics(element, {
        scrollWidth: 800,
        clientWidth: 300,
        scrollLeft: 0,
      });

      expect(readScrollFadeState(element)).toEqual({
        left: false,
        right: true,
      });
    });

    it("shows both fades in the middle of an overflowing table", () => {
      const element = document.createElement("div");
      mockScrollMetrics(element, {
        scrollWidth: 800,
        clientWidth: 300,
        scrollLeft: 250,
      });

      expect(readScrollFadeState(element)).toEqual({ left: true, right: true });
    });

    it("shows only the left fade at the end of an overflowing table", () => {
      const element = document.createElement("div");
      mockScrollMetrics(element, {
        scrollWidth: 800,
        clientWidth: 300,
        scrollLeft: 500,
      });

      expect(readScrollFadeState(element)).toEqual({
        left: true,
        right: false,
      });
    });
  });

  describe("MarkdownTableScroll", () => {
    beforeEach(() => {
      vi.stubGlobal(
        "ResizeObserver",
        class {
          observe = vi.fn();

          unobserve = vi.fn();

          disconnect = vi.fn();
        },
      );
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("renders animated edge fades that toggle with horizontal scroll", () => {
      render(
        <MarkdownTableScroll>
          <table>
            <tbody>
              <tr>
                <td>Wide content</td>
              </tr>
            </tbody>
          </table>
        </MarkdownTableScroll>,
      );

      const scroller = screen.getByTestId("markdown-table-scroll");
      const leftFade = screen.getByTestId("markdown-table-scroll-fade-left");
      const rightFade = screen.getByTestId("markdown-table-scroll-fade-right");

      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 0,
      });
      fireEvent.scroll(scroller);

      expect(rightFade).toHaveAttribute("data-visible", "true");
      expect(rightFade).toHaveClass("opacity-100");
      expect(leftFade).toHaveAttribute("data-visible", "false");
      expect(leftFade).toHaveClass("opacity-0");

      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 580,
      });
      fireEvent.scroll(scroller);

      expect(leftFade).toHaveAttribute("data-visible", "true");
      expect(leftFade).toHaveClass("opacity-100");
      expect(rightFade).toHaveAttribute("data-visible", "false");
      expect(rightFade).toHaveClass("opacity-0");
    });
  });
});

describe("plan-components", () => {
  describe("planComponents", () => {
    describe("h1", () => {
      it("should render h1 with correct text content", () => {
        // Arrange
        const H1 = planComponents.h1;
        const text = "Main Heading";

        // Act
        render(<H1>{text}</H1>);

        // Assert
        const heading = screen.getByRole("heading", { level: 1 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });

      it("should handle undefined children gracefully", () => {
        // Arrange
        const H1 = planComponents.h1;

        // Act
        render(<H1>{undefined}</H1>);

        // Assert
        const heading = screen.getByRole("heading", { level: 1 });
        expect(heading).toBeInTheDocument();
        expect(heading).toBeEmptyDOMElement();
      });

      it("should render complex children content", () => {
        // Arrange
        const H1 = planComponents.h1;

        // Act
        render(
          <H1>
            <span>Nested</span> Content
          </H1>,
        );

        // Assert
        const heading = screen.getByRole("heading", { level: 1 });
        expect(heading).toHaveTextContent("Nested Content");
        expect(heading.querySelector("span")).toHaveTextContent("Nested");
      });
    });

    describe("h2", () => {
      it("should render h2 with correct text content", () => {
        // Arrange
        const H2 = planComponents.h2;
        const text = "Section Heading";

        // Act
        render(<H2>{text}</H2>);

        // Assert
        const heading = screen.getByRole("heading", { level: 2 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });

      it("should handle null children gracefully", () => {
        // Arrange
        const H2 = planComponents.h2;

        // Act
        render(<H2>{null}</H2>);

        // Assert
        const heading = screen.getByRole("heading", { level: 2 });
        expect(heading).toBeInTheDocument();
        expect(heading).toBeEmptyDOMElement();
      });
    });

    describe("h3", () => {
      it("should render h3 with correct text content", () => {
        // Arrange
        const H3 = planComponents.h3;
        const text = "Subsection Heading";

        // Act
        render(<H3>{text}</H3>);

        // Assert
        const heading = screen.getByRole("heading", { level: 3 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });
    });

    describe("h4", () => {
      it("should render h4 with correct text content", () => {
        // Arrange
        const H4 = planComponents.h4;
        const text = "Level 4 Heading";

        // Act
        render(<H4>{text}</H4>);

        // Assert
        const heading = screen.getByRole("heading", { level: 4 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });
    });

    describe("h5", () => {
      it("should render h5 with correct text content", () => {
        // Arrange
        const H5 = planComponents.h5;
        const text = "Level 5 Heading";

        // Act
        render(<H5>{text}</H5>);

        // Assert
        const heading = screen.getByRole("heading", { level: 5 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });
    });

    describe("h6", () => {
      it("should render h6 with correct text content", () => {
        // Arrange
        const H6 = planComponents.h6;
        const text = "Level 6 Heading";

        // Act
        render(<H6>{text}</H6>);

        // Assert
        const heading = screen.getByRole("heading", { level: 6 });
        expect(heading).toBeInTheDocument();
        expect(heading).toHaveTextContent(text);
      });
    });

    describe("heading hierarchy", () => {
      it("should render all heading levels correctly in sequence", () => {
        // Arrange
        const H1 = planComponents.h1;
        const H2 = planComponents.h2;
        const H3 = planComponents.h3;
        const H4 = planComponents.h4;
        const H5 = planComponents.h5;
        const H6 = planComponents.h6;

        // Act
        render(
          <div>
            <H1>Heading 1</H1>
            <H2>Heading 2</H2>
            <H3>Heading 3</H3>
            <H4>Heading 4</H4>
            <H5>Heading 5</H5>
            <H6>Heading 6</H6>
          </div>,
        );

        // Assert
        expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
          "Heading 1",
        );
        expect(screen.getByRole("heading", { level: 2 })).toHaveTextContent(
          "Heading 2",
        );
        expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent(
          "Heading 3",
        );
        expect(screen.getByRole("heading", { level: 4 })).toHaveTextContent(
          "Heading 4",
        );
        expect(screen.getByRole("heading", { level: 5 })).toHaveTextContent(
          "Heading 5",
        );
        expect(screen.getByRole("heading", { level: 6 })).toHaveTextContent(
          "Heading 6",
        );
      });
    });

    describe("p", () => {
      it("should render paragraph with correct text content", () => {
        // Arrange
        const P = planComponents.p;
        const text = "Paragraph text";

        // Act
        render(<P>{text}</P>);

        // Assert
        expect(screen.getByText(text)).toBeInTheDocument();
      });
    });

    describe("ul", () => {
      it("should render unordered list with correct content", () => {
        // Arrange
        const Ul = planComponents.ul;
        const Li = planComponents.li;

        // Act
        render(
          <Ul>
            <Li>Item 1</Li>
            <Li>Item 2</Li>
          </Ul>,
        );

        // Assert
        expect(screen.getByRole("list")).toBeInTheDocument();
        expect(screen.getAllByRole("listitem")).toHaveLength(2);
      });
    });

    describe("ol", () => {
      it("should render ordered list with correct content", () => {
        // Arrange
        const Ol = planComponents.ol;
        const Li = planComponents.li;

        // Act
        render(
          <Ol>
            <Li>Item 1</Li>
            <Li>Item 2</Li>
          </Ol>,
        );

        // Assert
        expect(screen.getByRole("list")).toBeInTheDocument();
        expect(screen.getAllByRole("listitem")).toHaveLength(2);
      });

      it("should support start attribute", () => {
        // Arrange
        const Ol = planComponents.ol;
        const Li = planComponents.li;

        // Act
        const { container } = render(
          <Ol start={5}>
            <Li>Item 1</Li>
          </Ol>,
        );

        // Assert
        const ol = container.querySelector("ol");
        expect(ol).toHaveAttribute("start", "5");
      });
    });

    describe("a", () => {
      it("should render anchor with correct href and content", () => {
        // Arrange
        const A = planComponents.a;
        const href = "https://example.com";
        const text = "Link text";

        // Act
        render(<A href={href}>{text}</A>);

        // Assert
        const link = screen.getByRole("link");
        expect(link).toHaveTextContent(text);
        expect(link).toHaveAttribute("href", href);
        expect(link).toHaveAttribute("target", "_blank");
        expect(link).toHaveAttribute("rel", "noopener noreferrer");
      });
    });

    describe("code", () => {
      it("should render inline code with correct content", () => {
        // Arrange
        const Code = planComponents.code;
        const text = "const x = 1";

        // Act
        render(<Code>{text}</Code>);

        // Assert
        expect(screen.getByText(text)).toBeInTheDocument();
      });
    });
  });

  describe("createPlanComponents", () => {
    it("should apply extraClassName to all elements when provided", () => {
      // Arrange
      const extraClassName = "test-extra-class";
      const components = createPlanComponents(extraClassName);

      // Act
      const { container } = render(
        <div>
          <components.h1>H1</components.h1>
          <components.p>Paragraph</components.p>
          <components.ul>
            <components.li>Item</components.li>
          </components.ul>
          <components.a href="#">Link</components.a>
          <components.code>Code</components.code>
        </div>,
      );

      // Assert
      expect(container.querySelector("h1")).toHaveClass(extraClassName);
      expect(container.querySelector("p")).toHaveClass(extraClassName);
      expect(container.querySelector("ul")).toHaveClass(extraClassName);
      expect(container.querySelector("li")).toHaveClass(extraClassName);
      expect(container.querySelector("a")).toHaveClass(extraClassName);
      expect(container.querySelector("code")).toHaveClass(extraClassName);
    });

    it("should not add undefined class when extraClassName is not provided", () => {
      // Arrange
      const components = createPlanComponents();

      // Act
      const { container } = render(<components.h1>H1</components.h1>);

      // Assert
      const h1 = container.querySelector("h1");
      expect(h1?.className).not.toContain("undefined");
    });
  });
});
