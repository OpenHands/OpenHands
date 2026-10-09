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

const probe = `
import { Divider } from "#/ui/divider";
import { Divider as AliasedDivider } from "#/ui/divider";
import { ToggleSwitch, ToggleSwitchVisual } from "#/ui/toggle-switch";
import { ToggleSwitch as ReExportedToggle } from "#/components/features/automations/toggle-switch";

export function Probe() {
  return <>
    {/* Allowed: layout classes on Divider, ToggleSwitch, and ToggleSwitchVisual */}
    <Divider className="mt-4 w-full" />
    <ToggleSwitch isToggled={false} onToggle={() => {}} className="mt-2 w-auto" />
    <ToggleSwitchVisual isToggled={false} className="inline-block" />

    {/* Allowed: opacity on ToggleSwitch button wrapper */}
    <ToggleSwitch isToggled={true} onToggle={() => {}} className="opacity-50" />
    <ReExportedToggle isToggled={true} onToggle={() => {}} className="opacity-75" />

    {/* Rejected: appearance, shape, internal spacing on Divider */}
    <Divider className="bg-primary" />
    <Divider className="rounded-lg" />
    <Divider className="p-4" />
    <AliasedDivider className="bg-primary" />
    <Divider className="opacity-50" />

    {/* Rejected: appearance, internal spacing on ToggleSwitch and Visual */}
    <ToggleSwitch isToggled={false} onToggle={() => {}} className="bg-primary" />
    <ToggleSwitch isToggled={false} onToggle={() => {}} className="p-4" />
    <ReExportedToggle isToggled={false} onToggle={() => {}} className="p-4" />
    <ToggleSwitchVisual isToggled={false} className="opacity-50" />
    <ToggleSwitchVisual isToggled={false} className="bg-primary" />
    <ToggleSwitchVisual isToggled={false} className="rounded-full" />
  </>;
}
`;

describe("Canvas scoped no-restyle lint policy", () => {
  it("rejects appearance, color, shape, and internal spacing while permitting layout and wrapper opacity", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/routes/no-restyle-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      (message) => message.ruleId === "shadcn/no-restyle",
    );
    expect(findings.length).toBe(11);
  });

  it("exempts primitive implementations in src/ui/** from no-restyle", async () => {
    const [result] = await eslint.lintText(probe, {
      filePath: "src/ui/no-restyle-probe.tsx",
    });
    expect(result.fatalErrorCount).toBe(0);
    const findings = result.messages.filter(
      (message) => message.ruleId === "shadcn/no-restyle",
    );
    expect(findings).toEqual([]);
  });
});
