import { render } from "@testing-library/react";
import { Trans } from "react-i18next";
import { describe, it, expect, beforeAll } from "vitest";
import i18n from "#/i18n";

/**
 * Regression: i18next's default ``escapeValue: true`` double-escaped
 * interpolated values because React's text renderer also escapes —
 * producing entity-encoded text like ``&#x2F;tmp&#x2F;foo`` in the DOM.
 * ``index.ts`` flips it to ``false``; this test guards against accidental
 * reversion.
 */
describe("i18next interpolation", () => {
  beforeAll(async () => {
    // Register a tiny test bundle so we don't depend on translation.json.
    i18n.addResourceBundle(
      "en",
      "translation",
      { TEST_PATH_INTERPOLATION: "Reading {{path}}" },
      true,
      true,
    );
    await i18n.changeLanguage("en");
  });

  it("renders interpolated paths verbatim (no HTML-entity escaping)", () => {
    const { container } = render(
      <Trans
        i18nKey="TEST_PATH_INTERPOLATION"
        values={{ path: "/tmp/pr14227/part_037.diff" }}
      />,
    );
    expect(container.textContent).toBe(
      "Reading /tmp/pr14227/part_037.diff",
    );
    // Specifically: the forward slashes must not be encoded.
    expect(container.innerHTML).not.toContain("&#x2F;");
    expect(container.innerHTML).not.toContain("&#x2f;");
  });

  it("still renders potentially-dangerous characters as literal text (React-side safety boundary)", () => {
    const { container } = render(
      <Trans
        i18nKey="TEST_PATH_INTERPOLATION"
        values={{ path: "<script>alert(1)</script>" }}
      />,
    );
    // React's text renderer turns ``<``/``>`` into entity references — they
    // appear as literal characters in textContent and do NOT execute.
    expect(container.textContent).toContain("<script>alert(1)</script>");
    expect(container.querySelector("script")).toBeNull();
  });
});
