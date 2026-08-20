import { describe, expect, it } from "vitest";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { buildPluginLaunchPath } from "#/utils/plugin-launch-url";

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

  it("correctly encodes plugins with Unicode parameters", () => {
    // Arrange: plugin with Unicode characters
    const plugins: PluginSpec[] = [
      {
        source: "github:owner/repo",
        parameters: { task: "整理发布说明 🚀" },
      },
    ];

    // Act: build the path
    const path = buildPluginLaunchPath(plugins);
    const url = new URL(path, "http://localhost");
    
    // Manually decode the way launch.tsx does
    const decodedB64 = atob(url.searchParams.get("plugins") ?? "");
    const bytes = new Uint8Array(decodedB64.length);
    for (let i = 0; i < decodedB64.length; i++) {
      bytes[i] = decodedB64.charCodeAt(i);
    }
    const utf8Decoded = new TextDecoder().decode(bytes);
    const decoded = JSON.parse(utf8Decoded);

    // Assert: Unicode is preserved exactly
    expect(decoded).toEqual(plugins);
  });
});
