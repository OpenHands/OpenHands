import { describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";

// The document shell's router-bound components (Meta, Links, Scripts,
// ScrollRestoration) need a data router; the shell contract under test is
// the static markup around them.
vi.mock("react-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router")>();
  return {
    ...actual,
    Meta: () => null,
    Links: () => null,
    Scripts: () => null,
    ScrollRestoration: () => null,
  };
});

import { Layout } from "#/root";

describe("document shell translation guard", () => {
  it("opts the document out of browser translation", () => {
    // #19755: a browser translation engine rewriting text nodes breaks
    // React's commit phase with repeating removeChild NotFoundErrors, and
    // Canvas ships its own language switcher, so the whole document opts
    // out.
    const html = renderToStaticMarkup(
      <Layout>
        <div>content</div>
      </Layout>,
    );
    expect(html).toMatch(/<html[^>]*translate="no"/);
    expect(html).toContain('<meta name="google" content="notranslate"');
  });
});
