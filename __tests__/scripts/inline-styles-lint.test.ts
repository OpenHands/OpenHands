// @vitest-environment node
import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

const eslint = new ESLint({
  overrideConfig: [
    {
      languageOptions: { parserOptions: { project: null } },
      rules: { "@typescript-eslint/prefer-optional-chain": "off" },
    },
  ],
});

const cases = [
  { jsx: '<div style={{ "--progress": width }} />', warning: false },
  {
    jsx: '<div style={{ "--ink": "var(--oh-foreground)" }} />',
    warning: false,
  },
  { jsx: "<div style={{ width }} />", warning: true },
  { jsx: "<div style={forwardedStyle} />", warning: true },
  { jsx: '<div style={{ "--ink": "#ff0000" }} />', warning: true },
  { jsx: '<style>{".probe { color: red; }"}</style>', warning: true },
];
const probe = `export function Probe({ width, forwardedStyle }) { return <>\n${cases.map(({ jsx }) => jsx).join("\n")}\n</>; }`;

describe("Canvas inline-style warning policy", () => {
  it("warns about inline styling while accepting dynamic variables and semantic tokens", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/routes/inline-style-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      ({ ruleId }) => ruleId === "shadcn/no-inline-styles",
    );
    expect(new Set(findings.map(({ line }) => line - 2))).toEqual(
      new Set(cases.flatMap(({ warning }, index) => (warning ? [index] : []))),
    );
    expect(findings.every(({ severity }) => severity === 1)).toBe(true);
  });

  it("preserves the embedding root's caller-owned style API", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/components/providers/agent-server-ui-root.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    expect(
      result.messages.filter(
        ({ ruleId }) => ruleId === "shadcn/no-inline-styles",
      ),
    ).toEqual([]);
  });
});
