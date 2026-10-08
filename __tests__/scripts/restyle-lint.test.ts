// @vitest-environment node
import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

// Use the shipped policy; virtual probes only opt out of type-aware parsing.
const eslint = new ESLint({
  overrideConfig: [
    {
      languageOptions: { parserOptions: { project: null } },
      rules: { "@typescript-eslint/prefer-optional-chain": "off" },
    },
  ],
});

const probe = `
import { Divider as Separator } from "#/ui/divider";
import { ToggleSwitch, ToggleSwitchVisual as Track } from "#/ui/toggle-switch";
import { ToggleSwitch as AutomationToggle } from "#/components/features/automations/toggle-switch";
import { Typography } from "#/ui/typography";
export function Probe() {
  return <>
    <Separator className="mt-2 w-full" />
    <Separator className="bg-danger rounded-md" />
    <Track enabled className="bg-danger" />
    <ToggleSwitch className="ml-2 opacity-50" />
    <ToggleSwitch className="bg-danger" />
    <AutomationToggle className="bg-danger" />
    <Track enabled className="opacity-50" />
    <Typography.Text className="text-danger" />
    <div className="bg-danger" />
  </>;
}
`;

describe("Canvas divider and toggle appearance contracts", () => {
  it("rejects restyling through aliases and re-exports while keeping allowed usage", async () => {
    // Layout and caller-owned wrapper opacity remain supported. Other Canvas
    // primitives and HTML are deliberately outside this first rollout.
    const [result] = await eslint.lintText(probe, {
      filePath: "src/routes/restyle-probe.tsx",
    });

    expect(result.fatalErrorCount).toBe(0);
    expect(
      result.messages
        .filter((message) => message.ruleId === "shadcn/no-restyle")
        .map(({ line, severity }) => ({ line, severity })),
    ).toEqual([
      { line: 9, severity: 2 },
      { line: 9, severity: 2 },
      { line: 10, severity: 2 },
      { line: 12, severity: 2 },
      { line: 13, severity: 2 },
      { line: 14, severity: 2 },
    ]);
  });

  it("allows primitive implementations to own their appearance", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/ui/restyle-probe.tsx",
    });

    expect(result.fatalErrorCount).toBe(0);
    expect(
      result.messages.filter(
        (message) => message.ruleId === "shadcn/no-restyle",
      ),
    ).toEqual([]);
  });
});
