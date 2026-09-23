import { describe, expect, it } from "vitest";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  getPluginDisplayName,
  getPluginSourceLabel,
  isLocalPluginSource,
  pluginReferenceKey,
} from "#/utils/plugin-display";

function buildPlugin(overrides: Partial<PluginSpec> = {}): PluginSpec {
  return {
    source: "github:OpenHands/extensions",
    ref: null,
    repo_path: null,
    parameters: null,
    name: null,
    ...overrides,
  };
}

describe("getPluginDisplayName", () => {
  it("prefers an explicit display name over repository coordinates", () => {
    expect(
      getPluginDisplayName(
        buildPlugin({
          name: "Weather Plugin",
          repo_path: "plugins/weather",
          source: "github:example/fallback",
        }),
      ),
    ).toBe("Weather Plugin");
  });

  it.each([
    ["plugins/weather", "weather"],
    ["/plugins/weather/", "weather"],
    ["///plugins///weather///", "weather"],
  ])("uses the final non-empty repo-path segment from %s", (repoPath, name) => {
    expect(getPluginDisplayName(buildPlugin({ repo_path: repoPath }))).toBe(
      name,
    );
  });

  it.each([
    ["github:OpenHands/extensions", "OpenHands/extensions"],
    ["github:", ""],
    ["https://github.com/OpenHands/extensions.git", "extensions"],
    ["https://github.com/OpenHands/extensions", "extensions"],
    ["git+ssh://github.com/OpenHands/extensions.git", "extensions"],
    ["git@github.com:OpenHands/extensions.git", "extensions"],
    ["OpenHands/extensions.git", "extensions"],
    ["OpenHands/extensions.GIT", "extensions.GIT"],
    ["OpenHands/extensions.git.backup", "extensions.git.backup"],
    ["local-plugin", "local-plugin"],
    ["local-plugin.git", "local-plugin.git"],
    ["", ""],
  ])("derives %j from source %j", (source, expectedName) => {
    expect(getPluginDisplayName(buildPlugin({ source }))).toBe(expectedName);
  });

  it.each(["https://github.com/OpenHands/extensions/", "/", "///"])(
    "falls back to the full source when %j ends in an empty segment",
    (source) => {
      expect(getPluginDisplayName(buildPlugin({ source }))).toBe(source);
    },
  );

  it("falls through empty names and empty repo paths", () => {
    expect(
      getPluginDisplayName(
        buildPlugin({
          name: "",
          repo_path: "",
          source: "bare-source",
        }),
      ),
    ).toBe("bare-source");
  });

  it("falls through repo paths containing only separators", () => {
    expect(
      getPluginDisplayName(
        buildPlugin({
          repo_path: "///",
          source: "github:OpenHands/extensions",
        }),
      ),
    ).toBe("OpenHands/extensions");
  });
});

describe("isLocalPluginSource", () => {
  it.each([
    "github:OpenHands/extensions",
    "github:",
    "https://github.com/OpenHands/extensions.git",
    "HTTP://github.com/OpenHands/extensions.git",
    "git+ssh://github.com/OpenHands/extensions.git",
    "ssh://git@github.com/OpenHands/extensions.git",
    "file:///tmp/local-plugin",
    "a://host/repository",
    "z9+.-://host/repository",
  ])("classifies the remote coordinate %j as non-local", (source) => {
    expect(isLocalPluginSource(buildPlugin({ source }))).toBe(false);
  });

  it.each([
    "local",
    "/Users/me/plugins/weather",
    "./plugins/weather",
    "../plugins/weather",
    "bare-plugin",
    "",
    "GitHub:OpenHands/extensions",
    "1https://github.com/OpenHands/extensions",
    "git_ssh://github.com/OpenHands/extensions",
    "ssh:git@github.com/OpenHands/extensions",
  ])("classifies the non-URL coordinate %j as local", (source) => {
    expect(isLocalPluginSource(buildPlugin({ source }))).toBe(true);
  });
});

describe("getPluginSourceLabel", () => {
  it("removes the GitHub prefix and appends a ref", () => {
    expect(
      getPluginSourceLabel(
        buildPlugin({
          source: "github:OpenHands/extensions",
          ref: "main",
        }),
      ),
    ).toBe("OpenHands/extensions @ main");
  });

  it.each([undefined, null, ""])("omits a missing or empty ref (%j)", (ref) => {
    expect(
      getPluginSourceLabel(
        buildPlugin({ source: "github:OpenHands/extensions", ref }),
      ),
    ).toBe("OpenHands/extensions");
  });

  it("preserves non-GitHub source coordinates", () => {
    expect(
      getPluginSourceLabel(
        buildPlugin({
          source: "https://github.com/OpenHands/extensions.git",
          ref: "v1.2.3",
        }),
      ),
    ).toBe("https://github.com/OpenHands/extensions.git @ v1.2.3");
  });

  it("only removes github: when it prefixes the source coordinate", () => {
    expect(
      getPluginSourceLabel(
        buildPlugin({
          source: "mirror:github:OpenHands/extensions",
          ref: "main",
        }),
      ),
    ).toBe("mirror:github:OpenHands/extensions @ main");
  });

  it.each([
    ["github:", "main", " @ main"],
    ["", "main", " @ main"],
    ["bare-plugin", "0", "bare-plugin @ 0"],
  ])(
    "preserves boundary source and ref strings for %j at %j",
    (source, ref, expectedLabel) => {
      expect(getPluginSourceLabel(buildPlugin({ source, ref }))).toBe(
        expectedLabel,
      );
    },
  );
});

describe("pluginReferenceKey", () => {
  it("joins all plugin coordinates in a stable order", () => {
    expect(
      pluginReferenceKey(
        buildPlugin({
          source: "github:OpenHands/extensions",
          ref: "main",
          repo_path: "plugins/weather/",
        }),
      ),
    ).toBe("github:OpenHands/extensions main plugins/weather/");
  });

  it("normalizes null, undefined, and empty optional coordinates", () => {
    const withoutOptionals = pluginReferenceKey(
      buildPlugin({ ref: undefined, repo_path: undefined }),
    );

    expect(withoutOptionals).toBe("github:OpenHands/extensions  ");
    expect(
      pluginReferenceKey(buildPlugin({ ref: null, repo_path: null })),
    ).toBe(withoutOptionals);
    expect(pluginReferenceKey(buildPlugin({ ref: "", repo_path: "" }))).toBe(
      withoutOptionals,
    );
  });

  it("ignores display-only data and parameters", () => {
    const coordinates = {
      source: "github:OpenHands/extensions",
      ref: "main",
      repo_path: "plugins/weather",
    };

    expect(
      pluginReferenceKey(
        buildPlugin({
          ...coordinates,
          name: "Weather",
          parameters: { unit: "celsius" },
        }),
      ),
    ).toBe(
      pluginReferenceKey(
        buildPlugin({
          ...coordinates,
          name: "Renamed Weather",
          parameters: { unit: "fahrenheit" },
        }),
      ),
    );
  });

  it("changes when any coordinate changes", () => {
    const keys = [
      buildPlugin({ source: "github:one/plugin", ref: "main" }),
      buildPlugin({ source: "github:two/plugin", ref: "main" }),
      buildPlugin({ source: "github:one/plugin", ref: "next" }),
      buildPlugin({
        source: "github:one/plugin",
        ref: "main",
        repo_path: "plugins/child",
      }),
    ].map(pluginReferenceKey);

    expect(new Set(keys).size).toBe(keys.length);
  });

  it("does not trim coordinate boundary strings", () => {
    expect(
      pluginReferenceKey(
        buildPlugin({ source: " source ", ref: " ref ", repo_path: " path " }),
      ),
    ).toBe(" source   ref   path ");
    expect(pluginReferenceKey(buildPlugin({ source: "" }))).toBe("  ");
  });
});
