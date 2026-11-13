// @vitest-environment node
import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

// Exercise the shipped config, not a separate copy of the rule settings.
// Virtual TSX probes need syntax parsing but do not belong to tsconfig's project.
const eslint = new ESLint({
  overrideConfig: [
    {
      languageOptions: { parserOptions: { project: null } },
      rules: { "@typescript-eslint/prefer-optional-chain": "off" },
    },
  ],
});

const probe = `
import { Typography as Type } from "#/ui/typography";
import { ComboboxCaretIcon as Caret } from "#/ui/combobox-caret";
export function Probe({ color, wide, className }: { color: string; wide: boolean; className?: string }) {
  return <>
    <Type.Text className={\`text-\${color}\`} />
    <Caret className={unresolvedClasses()} />
    <Type.Text className={wide ? "w-full" : "w-auto"} />
    <Caret className={className} />
    <div className="flex-cols" />
  </>;
}
`;

describe("Canvas static class policy", () => {
  it("rejects opaque UI classes while permitting static choices and forwarding", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/routes/lint-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      (message) => message.ruleId === "shadcn/require-static-classes",
    );
    expect(findings.map(({ line, severity }) => ({ line, severity }))).toEqual([
      { line: 6, severity: 2 },
      { line: 7, severity: 2 },
    ]);
  });

  it("exempts primitive internals but still rejects unknown utilities there", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/ui/lint-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    expect(
      result.messages.filter(
        (message) => message.ruleId === "shadcn/require-static-classes",
      ),
    ).toEqual([]);
    expect(
      result.messages
        .filter((message) => message.ruleId === "shadcn/no-unknown-classes")
        .map(({ line, severity }) => ({ line, severity })),
    ).toEqual([{ line: 10, severity: 2 }]);
  });
});
