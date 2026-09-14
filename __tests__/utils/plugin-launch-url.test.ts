import { describe, expect, it } from "vitest";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  buildPluginLaunchPath,
  decodePluginsPayload,
} from "#/utils/plugin-launch-url";

const decodeLikeLaunchRoute = (path: string): PluginSpec[] => {
  const url = new URL(path, "http://localhost");
  return JSON.parse(decodePluginsPayload(url.searchParams.get("plugins") ?? ""));
};

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

    // Assert: it targets /launch and round-trips the coordinates without corruption.
    expect(url.pathname).toBe("/launch");
    expect(decodeLikeLaunchRoute(path)).toEqual(plugins);
  });

  it("round-trips non-Latin-1 parameter values without corruption", () => {
    // Arrange: a parameter value with CJK characters and an emoji — bytes that
    // plain btoa/atob cannot carry as Latin-1 binary strings.
    const plugins: PluginSpec[] = [
      {
        source: "https://github.com/example/plugin",
        parameters: { task: "整理发布说明 🚀" },
      },
    ];

    // Act
    const path = buildPluginLaunchPath(plugins);

    // Assert
    expect(decodeLikeLaunchRoute(path)).toEqual(plugins);
  });

  it("keeps legacy ASCII base64 payloads decodable", () => {
    // Arrange: ASCII JSON encoded with plain btoa semantics — identical bytes
    // under UTF-8, so pre-Unicode links keep working.
    const plugins: PluginSpec[] = [
      { source: "github:OpenHands/extensions", ref: null, repo_path: null },
    ];
    const asciiPayload = btoa(JSON.stringify(plugins));

    // Act & Assert
    expect(JSON.parse(decodePluginsPayload(asciiPayload))).toEqual(plugins);
  });
});
