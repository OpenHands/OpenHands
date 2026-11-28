import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";
import {
  OPENHANDS_EXTENSIONS_VERSION,
  OPENHANDS_TYPESCRIPT_CLIENT_VERSION,
} from "./client-source";
import { resolvePackageVersion } from "../../scripts/resolve-package-version.mjs";

const require = createRequire(import.meta.url);

describe("bundled package versions", () => {
  it("reports the resolved client and extensions package versions", () => {
    expect(OPENHANDS_TYPESCRIPT_CLIENT_VERSION).toBe(
      resolvePackageVersion(
        "@openhands/typescript-client",
        require.resolve("@openhands/typescript-client"),
      ),
    );
    expect(OPENHANDS_EXTENSIONS_VERSION).toBe(
      resolvePackageVersion(
        "@openhands/extensions",
        require.resolve("@openhands/extensions"),
      ),
    );
  });
});
