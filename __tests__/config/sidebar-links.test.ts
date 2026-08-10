import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getConfiguredSidebarLinks,
  resetConfiguredSidebarLinksCache,
} from "#/config/sidebar-links";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_SIDEBAR_LINKS__;
  resetConfiguredSidebarLinksCache();
});

describe("getConfiguredSidebarLinks", () => {
  it("returns an empty array when no config is present", () => {
    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("parses a valid entry from VITE_SIDEBAR_LINKS", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "enterprise", label: "Enterprise", url: "https://openhands.dev" },
      ]),
    );

    expect(getConfiguredSidebarLinks()).toEqual([
      {
        id: "enterprise",
        label: "Enterprise",
        url: "https://openhands.dev",
        icon: "external-link",
      },
    ]);
  });

  it("falls back to window.__AGENT_CANVAS_SIDEBAR_LINKS__ when VITE_SIDEBAR_LINKS is empty", () => {
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_SIDEBAR_LINKS__ = JSON.stringify([
      {
        id: "docs",
        label: "Docs",
        url: "https://docs.example.com",
        icon: "book-open",
      },
    ]);

    expect(getConfiguredSidebarLinks()).toEqual([
      {
        id: "docs",
        label: "Docs",
        url: "https://docs.example.com",
        icon: "book-open",
      },
    ]);
  });

  it("prefers VITE_SIDEBAR_LINKS over the injected window global", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "from-env", label: "From env", url: "https://a.example.com" },
      ]),
    );
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_SIDEBAR_LINKS__ = JSON.stringify([
      { id: "from-window", label: "From window", url: "https://b.example.com" },
    ]);

    expect(getConfiguredSidebarLinks().map((l) => l.id)).toEqual(["from-env"]);
  });

  it("memoizes the result across calls", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "cached", label: "Cached", url: "https://example.com" },
      ]),
    );

    const first = getConfiguredSidebarLinks();
    vi.stubEnv("VITE_SIDEBAR_LINKS", "");
    const second = getConfiguredSidebarLinks();

    expect(second).toBe(first);
  });

  it("drops the whole config when it is not valid JSON", () => {
    vi.stubEnv("VITE_SIDEBAR_LINKS", "{not json");
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("drops the whole config when it is not a JSON array", () => {
    vi.stubEnv("VITE_SIDEBAR_LINKS", JSON.stringify({ id: "not-an-array" }));
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("drops an entry missing a label", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([{ id: "no-label", url: "https://example.com" }]),
    );
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("drops an entry with a non-http(s) URL", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "js-url", label: "Bad", url: "javascript:alert(1)" },
      ]),
    );
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("drops an entry with an invalid id", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "Not Kebab Case!", label: "Bad", url: "https://example.com" },
      ]),
    );
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([]);
  });

  it("drops the second entry when two entries share an id, keeping the first", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "dup", label: "First", url: "https://a.example.com" },
        { id: "dup", label: "Second", url: "https://b.example.com" },
      ]),
    );
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks()).toEqual([
      {
        id: "dup",
        label: "First",
        url: "https://a.example.com",
        icon: "external-link",
      },
    ]);
  });

  it("falls back to the external-link icon for an unknown icon slug", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        {
          id: "unknown-icon",
          label: "Unknown",
          url: "https://example.com",
          icon: "rocket",
        },
      ]),
    );

    expect(getConfiguredSidebarLinks()[0]?.icon).toBe("external-link");
  });

  it("keeps valid entries alongside a dropped invalid one", () => {
    vi.stubEnv(
      "VITE_SIDEBAR_LINKS",
      JSON.stringify([
        { id: "ok", label: "OK", url: "https://example.com" },
        { id: "bad", label: "", url: "https://example.com" },
      ]),
    );
    vi.spyOn(console, "warn").mockImplementation(() => {});

    expect(getConfiguredSidebarLinks().map((l) => l.id)).toEqual(["ok"]);
  });
});
