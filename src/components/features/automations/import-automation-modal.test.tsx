import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { AutomationSpec } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";
import { ImportAutomationModal } from "./import-automation-modal";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/components/shared/modals/modal-backdrop", () => ({
  ModalBackdrop: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("#/components/shared/modals/modal-close-button", () => ({
  ModalCloseButton: () => (
    // eslint-disable-next-line i18next/no-literal-string
    <button type="button">close</button>
  ),
}));

const spec: AutomationSpec = {
  name: "Review new pull requests",
  prompt: "Review each new pull request and summarize any risks.",
  trigger: {
    type: "event",
    source: "github",
    on: "pull_request.opened",
  },
  enabled: true,
  plugins: ["github:openhands/extensions", "github:acme/review-tools"],
};

describe("ImportAutomationModal", () => {
  it("previews the parsed automation before import", () => {
    const markup = renderToStaticMarkup(
      <ImportAutomationModal
        isOpen
        spec={spec}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
      />,
    );

    expect(markup).toContain('data-testid="import-automation-modal"');
    expect(markup).toContain(spec.name);
    expect(markup).toContain("github: pull_request.opened");
    expect(markup).toContain(spec.prompt!);
    expect(markup).toContain(spec.plugins!.join(", "));
    expect(markup).toContain(I18nKey.AUTOMATIONS$IMPORT_DISABLED_NOTICE);
    expect(markup).toContain('data-testid="import-automation-confirm"');
  });

  it("does not render without a parsed spec", () => {
    const markup = renderToStaticMarkup(
      <ImportAutomationModal
        isOpen
        spec={null}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
      />,
    );

    expect(markup).toBe("");
  });
});
