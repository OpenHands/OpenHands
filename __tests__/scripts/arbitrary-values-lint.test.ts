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
const arbitrary =
  'export function Probe() { return <div className="w-[123px]" />; }';
const severities = async (source: string, filePath: string) => {
  const [result] = await eslint.lintText(source, { filePath });
  expect(result.fatalErrorCount).toBe(0);
  return result.messages
    .filter((message) => message.ruleId === "shadcn/no-arbitrary-values")
    .map((message) => message.severity);
};

describe("Canvas arbitrary-value guard adoption", () => {
  it.each([
    "src/components/features/command-menu/command-menu.tsx",
    "src/components/features/backends/backend-version.tsx",
    "src/components/features/settings/settings-nav-header.tsx",
    "src/components/features/settings/backend-synced-settings-badge.tsx",
  ])("rejects an arbitrary value in migrated owner %s", async (filePath) => {
    expect(await severities(arbitrary, filePath)).toEqual([2]);
  });

  it.each([
    "src/routes/geometry-probe.tsx",
    "src/ui/geometry-probe.tsx",
    "src/components/features/conversation/conversation-overview-panel.tsx",
  ])("retains warnings in unadopted owner %s", async (filePath) => {
    expect(await severities(arbitrary, filePath)).toEqual([1]);
  });

  it("accepts the adopted named recipes and tokens", async () => {
    expect(
      await severities(
        'export function Probe() { return <div className="command-menu-panel menu-caption-text text-2xs z-60" />; }',
        "src/components/features/command-menu/command-menu.tsx",
      ),
    ).toEqual([]);
  });
});
