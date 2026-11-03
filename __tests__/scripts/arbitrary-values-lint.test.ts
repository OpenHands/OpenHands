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

const adoptedOwners = [
  "src/components/features/command-menu/command-menu.tsx",
  "src/components/features/backends/backend-version.tsx",
  "src/components/features/settings/settings-nav-header.tsx",
  "src/components/features/settings/backend-synced-settings-badge.tsx",
];

const unadoptedPaths = [
  "src/routes/settings.tsx",
  "src/ui/button.tsx",
  "src/components/features/conversation-panel/start-task-card/start-task-status-badge.tsx",
];

const arbitraryClassProbe = `
export function Probe() {
  return <div className="p-[13px]" />;
}
`;

const themeClassesProbe = `
export function Probe() {
  return <div className="bg-surface text-text-dim rounded-lg px-1.5" />;
}
`;

const arbitraryVariantProbe = `
export function Probe() {
  return (
    <div className="data-[state=open]:bg-surface [&_svg]:size-4 bg-contrast/[0.09] hover:bg-contrast/[0.05]" />
  );
}
`;

describe("Canvas arbitrary values policy", () => {
  it.each(adoptedOwners)(
    "reports severity 2 (error) for arbitrary classes in adopted owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(arbitraryClassProbe, { filePath });
      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings).toEqual([
        expect.objectContaining({
          severity: 2,
        }),
      ]);
    },
  );

  it.each(unadoptedPaths)(
    "reports severity 1 (warn) for arbitrary classes in unadopted path %s",
    async (filePath) => {
      const [result] = await eslint.lintText(arbitraryClassProbe, { filePath });
      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings).toEqual([
        expect.objectContaining({
          severity: 1,
        }),
      ]);
    },
  );

  it.each(adoptedOwners)(
    "accepts named theme classes and recipes in adopted owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(themeClassesProbe, { filePath });
      expect(result.fatalErrorCount).toBe(0);
      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-arbitrary-values",
        ),
      ).toEqual([]);
    },
  );

  it.each(adoptedOwners)(
    "accepts arbitrary variant syntax and slash opacity modifiers in adopted owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(arbitraryVariantProbe, {
        filePath,
      });
      expect(result.fatalErrorCount).toBe(0);
      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-arbitrary-values",
        ),
      ).toEqual([]);
    },
  );

  it.each(adoptedOwners)(
    "produces zero arbitrary-value findings on actual owner file %s",
    async (filePath) => {
      const [result] = await eslint.lintFiles([filePath]);
      expect(result.fatalErrorCount).toBe(0);
      expect(
        result.messages.filter(
          (message) => message.ruleId === "shadcn/no-arbitrary-values",
        ),
      ).toEqual([]);
    },
  );
});
