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

const migratedOwners = [
  "src/components/features/command-menu/command-menu.tsx",
  "src/components/features/backends/backend-version.tsx",
  "src/components/features/settings/settings-nav-header.tsx",
  "src/components/features/settings/backend-synced-settings-badge.tsx",
];

const unadoptedPaths = [
  "src/routes/arbitrary-probe.tsx",
  "src/ui/arbitrary-probe.tsx",
  "src/components/features/conversation-panel/start-task-card/start-task-status-badge.tsx",
];

const arbitraryProbe = `
export function Probe() {
  return <div className="p-[13px]" />;
}
`;

const themeProbe = `
export function Probe() {
  return <div className="bg-surface text-text-dim rounded-lg px-1.5" />;
}
`;

const variantProbe = `
export function Probe() {
  return <div className="data-[state=open]:opacity-100 [&_svg]:size-4" />;
}
`;

describe("Canvas arbitrary-values rollout", () => {
  it.each(migratedOwners)(
    "reports severity 2 (error) for arbitrary classes in migrated owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(arbitraryProbe, { filePath });

      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings.length).toBeGreaterThan(0);
      expect(findings.every((f) => f.severity === 2)).toBe(true);
    },
  );

  it.each(unadoptedPaths)(
    "reports severity 1 (warn) for arbitrary classes in unadopted path %s",
    async (filePath) => {
      const [result] = await eslint.lintText(arbitraryProbe, { filePath });

      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings.length).toBeGreaterThan(0);
      expect(findings.every((f) => f.severity === 1)).toBe(true);
    },
  );

  it.each(migratedOwners)(
    "accepts named theme classes in migrated owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(themeProbe, { filePath });

      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings).toEqual([]);
    },
  );

  it.each(migratedOwners)(
    "accepts arbitrary variant syntax in migrated owner %s",
    async (filePath) => {
      const [result] = await eslint.lintText(variantProbe, { filePath });

      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings).toEqual([]);
    },
  );

  it("produces zero shadcn/no-arbitrary-values findings across existing migrated owner files", async () => {
    const results = await eslint.lintFiles(migratedOwners);

    for (const result of results) {
      expect(result.fatalErrorCount).toBe(0);
      const findings = result.messages.filter(
        (message) => message.ruleId === "shadcn/no-arbitrary-values",
      );
      expect(findings).toEqual([]);
    }
  });
});
