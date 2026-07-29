import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";

const SRC_ROOT = join(process.cwd(), "src");
const EXCLUDED_SEGMENTS = new Set(["mocks", "routeTree.gen.ts"]);
const ALLOWED_AD_HOC_HTTP_FILES = new Set([
  "api/automation-service/automation-service.api.ts",
  "api/cloud/proxy.ts",
  "api/main-app-auth.ts",
]);

function collectSourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = join(dir, entry.name);
    // Normalize to forward slashes so the path matches ALLOWED_AD_HOC_HTTP_FILES
    // entries on Windows where path.relative() returns backslash-separated paths.
    const relPath = relative(SRC_ROOT, fullPath).replace(/\\/g, "/");

    if (entry.isDirectory()) {
      if (EXCLUDED_SEGMENTS.has(entry.name)) return [];
      return collectSourceFiles(fullPath);
    }

    if (EXCLUDED_SEGMENTS.has(entry.name)) return [];
    if (!/\.(ts|tsx)$/.test(entry.name)) return [];
    if (/\.(test|spec)\.(ts|tsx)$/.test(entry.name)) return [];
    return [relPath];
  });
}

describe("agent-server API access", () => {
  it("uses typed @openhands/typescript-client access instead of ad-hoc HTTP", () => {
    const violations = collectSourceFiles(SRC_ROOT).flatMap((relPath) => {
      const source = readFileSync(join(SRC_ROOT, relPath), "utf8");
      const fileViolations: string[] = [];

      if (/openHands\s*\./.test(source)) {
        fileViolations.push("uses the shared axios instance directly");
      }

      if (/\bcreateHttpClient\s*\(/.test(source)) {
        fileViolations.push("uses createHttpClient directly");
      }

      if (
        /from\s+["']@openhands\/typescript-client\/client\/http-client["']/.test(
          source,
        )
      ) {
        fileViolations.push("imports the low-level SDK HttpClient directly");
      }

      if (/\bnew\s+HttpClient\s*\(/.test(source)) {
        fileViolations.push("constructs HttpClient directly");
      }

      if (
        (/\baxios\s*\(/.test(source) ||
          /\baxios\s*\.\s*(?:create|get|post|put|patch|delete|request)\s*\(/.test(
            source,
          )) &&
        !ALLOWED_AD_HOC_HTTP_FILES.has(relPath)
      ) {
        fileViolations.push("uses axios directly for HTTP calls");
      }

      if (/\bfetch\s*\([\s\S]{0,200}['"`]\/api\//.test(source)) {
        fileViolations.push("calls an /api path with fetch directly");
      }

      return fileViolations.map((violation) => `${relPath}: ${violation}`);
    });

    expect(violations).toEqual([]);
  });
});
