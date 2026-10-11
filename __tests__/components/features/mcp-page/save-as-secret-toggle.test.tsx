import type { ReactNode } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SaveAsSecretToggle } from "#/components/features/mcp-page/save-as-secret-toggle";
import { COLOR_THEMES } from "#/themes/color-themes";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";

// HeroUI's Tooltip (used inside StyledTooltip) only mounts its content on a
// real hover event, which jsdom doesn't fire. Stub it so the content renders
// eagerly and is queryable in tests.
vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    content,
    children,
  }: {
    content: ReactNode;
    children: ReactNode;
  }) => (
    <>
      {children}
      <span data-testid="styled-tooltip-content">{content}</span>
    </>
  ),
}));

describe("SaveAsSecretToggle", () => {
  // ── rendering ──────────────────────────────────────────────────────────────

  it("attaches data-testid to the label using fieldKey", () => {
    render(
      <SaveAsSecretToggle
        fieldKey="MY_KEY"
        checked={false}
        onToggle={vi.fn()}
      />,
    );
    expect(
      screen.getByTestId("mcp-install-save-secret-MY_KEY"),
    ).toBeInTheDocument();
  });

  it("renders the field key inside a <code> element", () => {
    render(
      <SaveAsSecretToggle
        fieldKey="MY_KEY"
        checked={false}
        onToggle={vi.fn()}
      />,
    );
    const codeEl = screen.getByText("MY_KEY");
    expect(codeEl.tagName.toLowerCase()).toBe("code");
  });

  it("renders the checkbox unchecked when checked=false", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it("renders the checkbox checked when checked=true", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={true} onToggle={vi.fn()} />,
    );
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  // ── accessibility ──────────────────────────────────────────────────────────

  it("the info button has an aria-label describing its purpose", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // t(key) => key in tests, so aria-label equals the raw i18n key.
    const infoBtn = screen.getByRole("button");
    expect(infoBtn).toHaveAttribute("aria-label", "MCP$SAVE_AS_SECRET_TOOLTIP");
  });

  it("the visual track is hidden from the accessibility tree (aria-hidden)", () => {
    const { container } = render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // The decorative <span> that forms the visual slider track.
    const track = container.querySelector("span[aria-hidden='true']");
    expect(track).toBeInTheDocument();
  });

  // ── interaction ────────────────────────────────────────────────────────────

  it("calls onToggle(true) when the unchecked checkbox is clicked", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it("calls onToggle(false) when the checked checkbox is clicked", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={true} onToggle={onToggle} />,
    );
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onToggle).toHaveBeenCalledWith(false);
  });

  // ── tooltip ────────────────────────────────────────────────────────────────

  it("passes tooltip text to StyledTooltip as its content prop", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    // The mock renders StyledTooltip's content prop into a <span>.
    // t(key) => key, so the rendered text is the raw i18n key.
    expect(screen.getByTestId("styled-tooltip-content")).toHaveTextContent(
      "MCP$SAVE_AS_SECRET_TOOLTIP",
    );
  });

  // ── styling, theme contrast & overflow prevention ──────────────────────────

  it("does not use raw green classes and reuses ToggleSwitchVisual tokens when checked", () => {
    const { container } = render(
      <SaveAsSecretToggle
        fieldKey="GITHUB_TOKEN"
        checked={true}
        onToggle={vi.fn()}
      />,
    );
    const label = screen.getByTestId("mcp-install-save-secret-GITHUB_TOKEN");
    expect(label.className).not.toContain("green-500");
    expect(label.className).toContain("border-contrast");
    expect(label.className).toContain("bg-contrast");

    const track = container.querySelector("span[aria-hidden='true']");
    expect(track?.className).not.toContain("green-500");
    expect(track?.className).toContain("bg-contrast");

    const thumb = track?.querySelector("span");
    expect(thumb?.className).not.toContain("bg-white");
    expect(thumb?.className).toContain("bg-base-secondary");

    const code = screen.getByText("GITHUB_TOKEN");
    expect(code.className).not.toContain("green-500");
    expect(code.className).toContain("text-contrast");
  });

  it("provides visible keyboard focus indicators and truncates long secret names", () => {
    render(
      <SaveAsSecretToggle
        fieldKey="VERY_LONG_SECRET_IDENTIFIER_THAT_CAN_OVERFLOW_ON_MOBILE"
        checked={false}
        onToggle={vi.fn()}
      />,
    );
    const label = screen.getByTestId(
      "mcp-install-save-secret-VERY_LONG_SECRET_IDENTIFIER_THAT_CAN_OVERFLOW_ON_MOBILE",
    );
    expect(label.className).toContain("has-[:focus-visible]");

    const code = screen.getByText(
      "VERY_LONG_SECRET_IDENTIFIER_THAT_CAN_OVERFLOW_ON_MOBILE",
    );
    expect(code.className).toContain("truncate");
    expect(code.className).toContain("min-w-0");
  });

  it("uses readable contrast tokens for the secret name and help glyph across states", () => {
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={vi.fn()} />,
    );
    const code = screen.getByText("KEY");
    expect(code.className).not.toContain("text-tertiary-alt");
    expect(code.className).toContain("text-contrast");

    const helpBtn = screen.getByRole("button");
    expect(helpBtn.className).not.toContain("text-tertiary-alt");
    expect(helpBtn.className).toContain("text-contrast");
  });

  it("clicking the help button does not invoke onToggle", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={onToggle} />,
    );
    const helpBtn = screen.getByRole("button");
    fireEvent.click(helpBtn);
    expect(onToggle).not.toHaveBeenCalled();
  });

  it("clicking the row label toggles the checkbox and calls onToggle", () => {
    const onToggle = vi.fn();
    render(
      <SaveAsSecretToggle fieldKey="KEY" checked={false} onToggle={onToggle} />,
    );
    const label = screen.getByTestId("mcp-install-save-secret-KEY");
    fireEvent.click(label);
    expect(onToggle).toHaveBeenCalledWith(true);
  });

  // ── 5 built-in palettes contrast validation ────────────────────────────────

  describe("WCAG contrast compliance across built-in palettes", () => {
    function parseHex(hex: string): [number, number, number] {
      const h = hex.replace("#", "");
      if (h.length === 3) {
        return [
          Number.parseInt(h[0] + h[0], 16),
          Number.parseInt(h[1] + h[1], 16),
          Number.parseInt(h[2] + h[2], 16),
        ];
      }
      return [
        Number.parseInt(h.slice(0, 2), 16),
        Number.parseInt(h.slice(2, 4), 16),
        Number.parseInt(h.slice(4, 6), 16),
      ];
    }

    function luminance([r, g, b]: [number, number, number]): number {
      const channels = [r, g, b].map((c) => {
        const channel = c / 255;
        return channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
    }

    function mix(
      fgHex: string,
      alpha: number,
      bgHex: string,
    ): [number, number, number] {
      const [r1, g1, b1] = parseHex(fgHex);
      const [r2, g2, b2] = parseHex(bgHex);
      return [
        Math.round(r1 * alpha + r2 * (1 - alpha)),
        Math.round(g1 * alpha + g2 * (1 - alpha)),
        Math.round(b1 * alpha + b2 * (1 - alpha)),
      ];
    }

    function contrast(
      rgb1: [number, number, number],
      rgb2: [number, number, number],
    ): number {
      const l1 = luminance(rgb1);
      const l2 = luminance(rgb2);
      return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    }

    it.each(Object.entries(COLOR_THEMES))(
      "meets >= 4.5:1 text/glyph and >= 3:1 switch contrast on %s",
      (themeKey, theme) => {
        const variables: Record<string, string> = {
          ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
          ...theme.scale,
          ...theme.tokens,
        };

        const resolve = (name: string): string => {
          const val = variables[name];
          expect(val, `Missing theme variable ${name}`).toBeDefined();
          const match = /^var\((--[\w-]+)\)$/.exec(val);
          return match ? resolve(match[1]) : val;
        };

        const baseSecondary = resolve("--oh-color-base-secondary");
        const contrastHex = resolve("--oh-contrast");

        const bgUnchecked = parseHex(baseSecondary);
        const bgChecked = mix(contrastHex, 0.05, baseSecondary);
        const codeBgChecked = mix(
          contrastHex,
          0.05,
          `#${bgChecked.map((x) => x.toString(16).padStart(2, "0")).join("")}`,
        );

        const contrastRgb = parseHex(contrastHex);

        // 1. Row label meets >= 4.5:1 in both unchecked and checked states
        expect(
          contrast(contrastRgb, bgUnchecked),
          `${themeKey} row label unchecked`,
        ).toBeGreaterThanOrEqual(4.5);
        expect(
          contrast(contrastRgb, bgChecked),
          `${themeKey} row label checked`,
        ).toBeGreaterThanOrEqual(4.5);

        // 2. Secret name code meets >= 4.5:1 in both unchecked and checked states
        expect(
          contrast(contrastRgb, bgUnchecked),
          `${themeKey} secret name unchecked`,
        ).toBeGreaterThanOrEqual(4.5);
        expect(
          contrast(contrastRgb, codeBgChecked),
          `${themeKey} secret name checked`,
        ).toBeGreaterThanOrEqual(4.5);

        // 3. Help glyph meets >= 4.5:1 in both unchecked and checked states
        expect(
          contrast(contrastRgb, bgUnchecked),
          `${themeKey} help glyph unchecked`,
        ).toBeGreaterThanOrEqual(4.5);
        expect(
          contrast(contrastRgb, bgChecked),
          `${themeKey} help glyph checked`,
        ).toBeGreaterThanOrEqual(4.5);

        // 4. Switch thumb vs track meets >= 3:1
        // In checked state, track is contrast and thumb is base-secondary
        expect(
          contrast(parseHex(baseSecondary), contrastRgb),
          `${themeKey} switch thumb vs track`,
        ).toBeGreaterThanOrEqual(3.0);
      },
    );
  });
});
