import { describe, expect, it } from "vitest";
import {
  renderSkinAppSkill,
  sanitizeIconName,
  sanitizeTheme,
  satisfiesCanvasVersion,
  stripSecretValues,
  themeCss,
  themeVars,
  SKIN_APP_SKILL_NAME,
} from "../scripts/skin-service.mjs";

describe("renderSkinAppSkill", () => {
  it("wraps a bare markdown body in always-active frontmatter", () => {
    const out = renderSkinAppSkill("# My Skin\n\nDoes things.", "My Skin");
    expect(out).toMatch(/^---\nname: skin-app\n/);
    expect(out).toContain('description: What the "My Skin" skin');
    expect(out).toContain("# My Skin\n\nDoes things.");
    expect(out).not.toContain("triggers:");
  });

  it("replaces existing frontmatter but keeps its description", () => {
    const src =
      "---\nname: whatever\ndescription: Monitors Datadog.\ntriggers:\n  - datadog\n---\n\n# Body\n";
    const out = renderSkinAppSkill(src, "Datadog Monitor");
    expect(out).toMatch(/^---\nname: skin-app\n/);
    expect(out).toContain("description: Monitors Datadog.");
    expect(out).not.toContain("triggers:");
    expect(out).not.toContain("name: whatever");
    expect(out).toContain("# Body");
  });

  it("tolerates malformed frontmatter", () => {
    const out = renderSkinAppSkill("---\n: : :\n---\n# Body", null);
    expect(out).toContain(`name: ${SKIN_APP_SKILL_NAME}`);
    expect(out).toContain("# Body");
  });
});

describe("sanitizeIconName", () => {
  it("accepts kebab-case lucide icon names, normalized", () => {
    expect(sanitizeIconName("activity")).toBe("activity");
    expect(sanitizeIconName("bar-chart-3")).toBe("bar-chart-3");
    expect(sanitizeIconName("  Layout-Dashboard ")).toBe("layout-dashboard");
  });

  it("rejects anything that is not a plain icon name", () => {
    expect(sanitizeIconName(undefined)).toBeNull();
    expect(sanitizeIconName(null)).toBeNull();
    expect(sanitizeIconName("")).toBeNull();
    expect(sanitizeIconName("Bar Chart")).toBeNull();
    expect(sanitizeIconName("-leading")).toBeNull();
    expect(sanitizeIconName("trailing-")).toBeNull();
    expect(sanitizeIconName("a".repeat(65))).toBeNull();
    expect(sanitizeIconName("<script>")).toBeNull();
    expect(sanitizeIconName({ name: "activity" })).toBeNull();
  });
});

describe("sanitizeTheme", () => {
  it("returns null for missing/empty/non-object themes", () => {
    expect(sanitizeTheme(undefined)).toBeNull();
    expect(sanitizeTheme(null)).toBeNull();
    expect(sanitizeTheme({})).toBeNull();
    expect(sanitizeTheme("green")).toBeNull();
    expect(sanitizeTheme(["#fff"])).toBeNull();
  });

  it("keeps only known keys with valid hex colors, lowercased", () => {
    expect(
      sanitizeTheme({
        accent: "#39D98A",
        background: " #0b1614 ",
        evil: "#123456",
        surface: "url(javascript:alert(1))",
        text: "red",
      }),
    ).toEqual({ accent: "#39d98a", background: "#0b1614" });
  });

  it("accepts 3-digit hex", () => {
    expect(sanitizeTheme({ accent: "#0f0" })).toEqual({ accent: "#0f0" });
  });

  it("rejects css-injection attempts", () => {
    expect(sanitizeTheme({ accent: "#fff;} body{display:none" })).toBeNull();
  });
});

describe("themeVars", () => {
  it("returns null when there is no usable theme", () => {
    expect(themeVars(undefined)).toBeNull();
    expect(themeVars({ accent: "not-a-color" })).toBeNull();
  });

  it("anchors the grey scale on declared colors", () => {
    const vars = themeVars({
      background: "#0b1614",
      surface: "#11201c",
      text: "#e6efe9",
    })!;
    expect(vars["--cool-grey-950"]).toBe("#0b1614");
    expect(vars["--cool-grey-925"]).toBe("#11201c");
    expect(vars["--cool-grey-100"]).toBe("#e6efe9");
    // Derived steps are concrete hex values (mixed in sRGB) so they can be
    // re-encoded as HSL channels for HeroUI.
    expect(vars["--cool-grey-800"]).toMatch(/^#[0-9a-f]{6}$/);
    // HeroUI reads its own HSL-channel variables; the scale is mirrored there.
    expect(vars["--heroui-background"]).toMatch(/^[\d.]+ [\d.]+% [\d.]+%$/);
    expect(vars["--heroui-default-100"]).toBe(
      vars["--heroui-background"], // both anchored at grey-950
    );
    // No accent declared → accent variables untouched.
    expect(vars["--oh-accent"]).toBeUndefined();
  });

  it("maps accent to primary/logo/accent and defaults warning to it", () => {
    const vars = themeVars({ accent: "#39d98a", background: "#0b1614" })!;
    expect(vars["--oh-accent"]).toBe("#39d98a");
    expect(vars["--oh-color-primary"]).toBe("#39d98a");
    expect(vars["--oh-color-logo"]).toBe("#39d98a");
    expect(vars["--oh-warning"]).toBe("#39d98a");
    expect(vars["--oh-accent-foreground"]).toBe("#0b1614");
  });

  it("honors explicit status colors", () => {
    const vars = themeVars({
      accent: "#39d98a",
      success: "#4ade80",
      warning: "#fbbf24",
      danger: "#f87171",
    })!;
    expect(vars["--oh-success"]).toBe("#4ade80");
    expect(vars["--oh-warning"]).toBe("#fbbf24");
    expect(vars["--oh-danger"]).toBe("#f87171");
  });

  it("works from a minimal accent-only theme", () => {
    const vars = themeVars({ accent: "#39d98a" })!;
    expect(vars["--oh-accent"]).toBe("#39d98a");
    expect(vars["--cool-grey-950"]).toBeTruthy();
  });
});

describe("themeCss", () => {
  it("renders an empty :root when no theme", () => {
    expect(themeCss(undefined)).toContain(":root {}");
  });

  it("renders one declaration per derived variable", () => {
    const css = themeCss({ accent: "#39d98a", background: "#0b1614" });
    expect(css).toMatch(/^:root \{\n/);
    expect(css).toContain("--oh-accent: #39d98a;");
    expect(css).toContain("--cool-grey-950: #0b1614;");
  });
});

describe("satisfiesCanvasVersion", () => {
  it("accepts any version when the range is empty or *", () => {
    expect(satisfiesCanvasVersion("1.8.0", "")).toBe(true);
    expect(satisfiesCanvasVersion("1.8.0", undefined)).toBe(true);
    expect(satisfiesCanvasVersion("1.8.0", "*")).toBe(true);
  });

  it("enforces >= lower bounds", () => {
    expect(satisfiesCanvasVersion("1.8.0", ">=1.7.0")).toBe(true);
    expect(satisfiesCanvasVersion("1.6.9", ">=1.7.0")).toBe(false);
  });

  it("enforces combined ranges", () => {
    expect(satisfiesCanvasVersion("1.8.0", ">=1.7.0 <2.0.0")).toBe(true);
    expect(satisfiesCanvasVersion("2.0.0", ">=1.7.0 <2.0.0")).toBe(false);
  });

  it("never blocks dev builds", () => {
    expect(satisfiesCanvasVersion("dev", ">=1.7.0")).toBe(true);
    expect(satisfiesCanvasVersion("unknown", ">=99.0.0")).toBe(true);
  });

  it("accepts v-prefixed versions", () => {
    expect(satisfiesCanvasVersion("v1.8.0", ">=1.7.0")).toBe(true);
  });
});

describe("stripSecretValues", () => {
  it("removes secret-ish string values but keeps names/structure", () => {
    const input = {
      name: "slack",
      api_key: "sk-live-abc123",
      token: "xoxb-secret",
      url: "https://example.com",
      nested: { password: "hunter2", model: "gpt-5" },
    };
    expect(stripSecretValues(input)).toEqual({
      name: "slack",
      url: "https://example.com",
      nested: { model: "gpt-5" },
    });
  });

  it("removes masked values returned by the agent server", () => {
    expect(stripSecretValues({ value: "**********", kept: "yes" })).toEqual({
      kept: "yes",
    });
  });

  it("handles arrays and scalars", () => {
    expect(stripSecretValues([{ token: "x", a: 1 }, "plain"])).toEqual([
      { a: 1 },
      "plain",
    ]);
    expect(stripSecretValues("s")).toBe("s");
    expect(stripSecretValues(null)).toBe(null);
  });
});
