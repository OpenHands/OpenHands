import { describe, expect, it } from "vitest";
import {
  OPENGREP_SCAN_COMMAND,
  parseOpengrepJsonOutput,
} from "#/utils/opengrep-output";

describe("parseOpengrepJsonOutput", () => {
  it("maps opengrep JSON results into security findings", () => {
    const raw = JSON.stringify({
      results: [
        {
          check_id: "python.lang.security.audit.eval-detected",
          path: "src/app.py",
          start: { line: 10, col: 5 },
          end: { line: 10, col: 20 },
          extra: {
            message: "Detected use of eval",
            severity: "ERROR",
          },
        },
      ],
      errors: [],
      paths: { scanned: ["src/app.py"], skipped: [] },
    });

    const result = parseOpengrepJsonOutput(raw);

    expect(result.tool).toBe("opengrep");
    expect(result.findings).toHaveLength(1);
    expect(result.findings[0]).toMatchObject({
      ruleId: "python.lang.security.audit.eval-detected",
      message: "Detected use of eval",
      severity: "ERROR",
      filePath: "src/app.py",
      startLine: 10,
      startCol: 5,
      endLine: 10,
      endCol: 20,
    });
  });

  it("returns an empty findings list when results are missing", () => {
    const result = parseOpengrepJsonOutput(
      JSON.stringify({ errors: [], paths: { scanned: [], skipped: [] } }),
    );

    expect(result.findings).toEqual([]);
  });

  it("throws on invalid JSON", () => {
    expect(() => parseOpengrepJsonOutput("not-json")).toThrow("invalid_output");
  });
});

describe("OPENGREP_SCAN_COMMAND", () => {
  it("resolves opengrep and runs a JSON scan with auto config", () => {
    expect(OPENGREP_SCAN_COMMAND).toContain('command -v opengrep');
    expect(OPENGREP_SCAN_COMMAND).toContain('scan --json --config auto .');
    expect(OPENGREP_SCAN_COMMAND).toContain("opengrep_not_installed");
  });
});
