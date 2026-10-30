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
    <Separator className="bg-danger" />
    <ToggleSwitch className="ml-2 opacity-50" />
    <ToggleSwitch className="bg-danger" />
    <AutomationToggle className="bg-danger" />
    <Track enabled className="opacity-50" />
    <Typography.Text className="text-danger" />
    <div className="bg-danger" />
  </>;
}
`;

const caretCases = [
  { component: "Arrow", classes: "ml-2 size-4 text-muted", allowed: true },
  { component: "Arrow", classes: "rotate-90", allowed: true },
  { component: "StateArrow", classes: "mr-1 text-danger", allowed: true },
  {
    component: "Carets.ComboboxCaretButton",
    classes: "self-center",
    allowed: true,
  },
  { component: "Arrow", classes: "p-2", allowed: false },
  { component: "StateArrow", classes: "bg-danger", allowed: false },
  {
    component: "Carets.ComboboxCaretButton",
    classes: "rounded-md",
    allowed: false,
  },
  { component: "StateArrow", classes: "rotate-90", allowed: false },
  {
    component: "Carets.ComboboxCaretButton",
    classes: "hover:-rotate-90",
    allowed: false,
  },
];
const caretRows = caretCases.map(
  ({ component, classes }) =>
    `<${component} className=${JSON.stringify(classes)} />`,
);
const caretProbe = `import { ComboboxCaretIcon as Arrow, ComboboxCaretInline as StateArrow } from "#/ui/combobox-caret";
import * as Carets from "#/ui/combobox-caret";
export function CaretProbe() { return <>\n${caretRows.join("\n")}\n</>; }`;

describe("Canvas UI appearance contracts", () => {
  it("keeps caret layout and text color public while protecting appearance and state rotation", async () => {
    const [result] = await eslint.lintText(caretProbe, {
      filePath: "src/routes/caret-restyle-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      (message) => message.ruleId === "shadcn/no-restyle",
    );
    // Cases start on line 4. Compare behavior by row, without mirroring
    // fixture class strings or relying on diagnostic order/per-class counts.
    expect(new Set(findings.map(({ line }) => line - 4))).toEqual(
      new Set(
        caretCases.flatMap(({ allowed }, index) => (allowed ? [] : [index])),
      ),
    );
    expect(findings.every(({ severity }) => severity === 2)).toBe(true);
  });
  it("rejects restyling through aliases and re-exports while keeping allowed usage", async () => {
    // Layout and caller-owned wrapper opacity remain supported. Other Canvas
    // primitives and HTML are deliberately outside this first rollout.
    const [result] = await eslint.lintText(probe, {
      filePath: "src/routes/restyle-probe.tsx",
    });

    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      (message) => message.ruleId === "shadcn/no-restyle",
    );
    const probeLines = probe.split("\n");
    expect(
      new Set(findings.map(({ line }) => probeLines[line - 1]?.trim())),
    ).toEqual(
      new Set([
        '<Separator className="bg-danger" />',
        '<ToggleSwitch className="bg-danger" />',
        '<AutomationToggle className="bg-danger" />',
        '<Track enabled className="opacity-50" />',
      ]),
    );
    expect(findings.every(({ severity }) => severity === 2)).toBe(true);
  });

  it("allows primitive implementations to own their appearance", async () => {
    const [result] = await eslint.lintText(`${probe}\n${caretProbe}`, {
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
