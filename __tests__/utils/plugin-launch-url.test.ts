import { describe, expect, it } from "vitest";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  buildPluginLaunchPath,
  decodePluginsBase64,
} from "#/utils/plugin-launch-url";

describe("buildPluginLaunchPath", () => {
  it("encodes plugins into a /launch path that decodes back to the same specs", () => {
    // Arrange: coordinates whose JSON base64-encodes with URL-special chars (+ / =).
    const plugins: PluginSpec[] = [
      {
        source: "github:OpenHands/extensions",
        ref: "v1.2+3",
        repo_path: "sub/dir",
      },
    ];

    // Act: build the path, then decode the `plugins` param the way /launch does.
    const path = buildPluginLaunchPath(plugins);
    const url = new URL(path, "http://localhost");
    const decoded = JSON.parse(atob(url.searchParams.get("plugins") ?? ""));

    // Assert: it targets /launch and round-trips the coordinates without corruption.
    expect(url.pathname).toBe("/launch");
    expect(decoded).toEqual(plugins);
  });

  it("round-trips Unicode parameter values through the UTF-8 payload", () => {
    // Arrange: non-Latin-1 text makes plain btoa throw InvalidCharacterError
    // (#16735), and a UTF-8 payload decoded as Latin-1 corrupts the value.
    const plugins: PluginSpec[] = [
      {
        source: "github:example/plugin",
        parameters: { task: "整理发布说明 🚀" },
      },
    ];

    // Act + Assert: the builder must not throw, and the /launch decoder must
    // restore the exact Unicode value.
    const path = buildPluginLaunchPath(plugins);
    const url = new URL(path, "http://localhost");
    expect(decodePluginsBase64(url.searchParams.get("plugins") ?? "")).toEqual(
      plugins,
    );
  });

  it("decodes legacy ASCII payloads unchanged", () => {
    // Payload produced by the pre-UTF-8 builder for a simple ASCII spec.
    const legacy = btoa(
      JSON.stringify([{ source: "github:example/plugin", ref: null }]),
    );
    expect(decodePluginsBase64(legacy)).toEqual([
      { source: "github:example/plugin", ref: null },
    ]);
  });
});
