import { describe, expect, it } from "vitest";
import {
  SYFT_SCAN_COMMAND,
  encodeBomForDependencyTrack,
} from "#/utils/syft-output";

describe("SYFT_SCAN_COMMAND", () => {
  it("resolves syft and emits CycloneDX JSON for the workspace", () => {
    expect(SYFT_SCAN_COMMAND).toContain("command -v syft");
    expect(SYFT_SCAN_COMMAND).toContain('dir:. -o cyclonedx-json@1.5');
    expect(SYFT_SCAN_COMMAND).toContain("syft_not_installed");
  });
});

describe("encodeBomForDependencyTrack", () => {
  it("base64-encodes the SBOM JSON for Dependency-Track upload", () => {
    const bom = JSON.stringify({ bomFormat: "CycloneDX", specVersion: "1.5" });
    const encoded = encodeBomForDependencyTrack(bom);
    expect(encoded).toBe(btoa(bom));
    expect(JSON.parse(atob(encoded))).toEqual({
      bomFormat: "CycloneDX",
      specVersion: "1.5",
    });
  });
});
