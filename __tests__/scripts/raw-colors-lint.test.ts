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
