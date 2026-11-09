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
    "src/components/shared/modals/modal-backdrop.tsx",
    "src/components/features/backends/environment-switch-overlay.tsx",
    "src/components/features/chat/components/context-window-meter.tsx",
    "src/components/features/conversation-panel/conversation-panel-wrapper.tsx",
    "src/components/features/command-menu/command-menu.tsx",
    "src/components/features/backends/backend-version.tsx",
    "src/components/features/settings/settings-nav-header.tsx",
    "src/components/features/settings/backend-synced-settings-badge.tsx",
    "src/utils/custom-toast-handlers.tsx",
    "src/utils/form-control-classes.ts",
    "src/components/features/chat/chat-status-indicator.tsx",
    "src/components/features/chat/components/context-window-ring.tsx",
    "src/components/features/chat/user-message-body.tsx",
    "src/components/features/conversation/conversation-main/conversation-main.tsx",
    "src/components/features/chat/drag-over.tsx",
    "src/components/features/chat/chat-markdown-path-code.tsx",
    "src/components/features/markdown/code.tsx",
    "src/components/shared/horizontal-scroll-fade.tsx",
    "src/components/features/markdown/markdown-table-scroll.tsx",
    "src/components/features/automations/recommended-automations-rail.tsx",
    "src/components/features/automations/automation-view-toggle.tsx",
    "src/components/features/automations/dashboard/automations-dashboard-controls.tsx",
    "src/components/features/automations/detail/run-phase.tsx",
    "src/components/features/conversation-panel/conversation-card/conversation-card.tsx",
    "src/utils/hover-reveal-classes.ts",
  ])("rejects an arbitrary value in migrated owner %s", async (filePath) => {
    const source = filePath.endsWith(".ts")
      ? 'import { cn } from "#/utils/utils"; export const probe = cn("w-[123px]");'
      : arbitrary;
    expect(await severities(source, filePath)).toEqual([2]);
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
