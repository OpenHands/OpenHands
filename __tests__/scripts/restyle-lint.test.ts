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

describe("raw colors", () => {
  const probe = `
export function Probe() {
  return <>
    <div className="text-red-500" />
    <div className="text-unregistered-color" />
    <svg><path fill="#ec4899" /></svg>
    <div className="text-danger bg-surface text-foreground" />
    <div className="bg-background bg-content1 text-primary-foreground text-medium" />
    <div className="bg-status-fail-bg text-status-fail-text text-content-muted text-modal-muted" />
    <div className="bg-content9" />
  </>;
}
`;

  describe("Canvas raw color rollout", () => {
    it("warns on palette classes reached through an adopted dynamic map", async () => {
      const [result] = await eslint.lintText(
        `
const toneClasses = {
  warning: "text-yellow-500",
  muted: "text-muted",
};
export function Probe({ tone }: { tone: keyof typeof toneClasses }) {
  return <div className={toneClasses[tone]} />;
}
`,
        { filePath: "src/components/features/markdown/blockquote.tsx" },
      );

      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-raw-colors",
        ),
      ).toEqual([
        expect.objectContaining({
          severity: 1,
          messageId: expect.stringMatching(/^paletteClass/),
        }),
      ]);
    });

    it("keeps unrelated program tags outside broad string scanning", async () => {
      const [result] = await eslint.lintText(
        'export type Variant = "default" | "text-like";',
        { filePath: "src/components/shared/buttons/modal-button.tsx" },
      );

      expect(result.fatalErrorCount).toBe(0);
      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-raw-colors",
        ),
      ).toEqual([]);
    });

    it("accepts SVG classes that disable paint", async () => {
      const [result] = await eslint.lintText(
        'export function Probe() { return <svg className="fill-none stroke-none" />; }',
        { filePath: "src/routes/paint-probe.tsx" },
      );

      expect(result.fatalErrorCount).toBe(0);
      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-raw-colors",
        ),
      ).toEqual([]);
    });

    it.each(["src/routes/color-probe.tsx", "src/ui/color-probe.tsx"])(
      "warns on raw colors and unknown tokens while accepting verified tokens in %s",
      async (filePath) => {
        const [result] = await eslint.lintText(probe, { filePath });
        expect(result.fatalErrorCount).toBe(0);
        expect(
          result.messages
            .filter((message) => message.ruleId === "shadcn/no-raw-colors")
            .map(({ line, severity }) => ({ line, severity })),
        ).toEqual([
          { line: 4, severity: 1 },
          { line: 5, severity: 1 },
          { line: 6, severity: 1 },
          { line: 10, severity: 1 },
        ]);
      },
    );
  });
});

describe("static classes", () => {
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
      expect(
        findings.map(({ line, severity }) => ({ line, severity })),
      ).toEqual([
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
});

describe("inline styles", () => {
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
        new Set(
          cases.flatMap(({ warning }, index) => (warning ? [index] : [])),
        ),
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
});
