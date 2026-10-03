import assert from "node:assert/strict";
import { ESLint } from "eslint";

// Run from the repository root: node .pr/verify-unknown-classes.mjs
// Use the real flat config and theme, but let lintText accept a virtual TSX file.
const eslint = new ESLint({
  overrideConfig: [
    {
      languageOptions: { parserOptions: { project: null } },
      rules: { "@typescript-eslint/prefer-optional-chain": "off" },
    },
  ],
});
const classes = [
  "hovr:flex",
  "flex-cols", // Deliberate mistakes; also detect grammar fallback.
  "bg-surface-raised",
  "border-border-input",
  "ring-focus", // Canvas theme.
  "prose",
  "scrollbar-hide",
  "data-[hover=true]:bg-default-100", // Plugins.
  "environment-switch-overlay",
  "conversation-overview-diffs-git-action", // Exceptions.
  "environment-switch-overla", // A nearby typo must still be reported.
];
const [result] = await eslint.lintText(
  `export const probe = <div className="${classes.join(" ")}" />;`,
  { filePath: "src/ui/lint-probe.tsx" },
);
assert.equal(result.fatalErrorCount, 0);
const findings = result.messages.filter(
  (m) => m.ruleId === "shadcn/no-unknown-classes",
);
assert.deepEqual(findings.map((m) => m.message.match(/^"([^"]+)"/)[1]).sort(), [
  "environment-switch-overla",
  "flex-cols",
  "hovr:flex",
]);
for (const finding of findings) console.log(finding.message);
console.log(
  "PASS: real theme/plugins accepted; typos reported; exceptions stay exact.",
);
