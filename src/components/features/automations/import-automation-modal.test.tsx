import { fireEvent, render, screen } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { AutomationSpec } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";
import { AUTOMATION_FILE_FORMAT_DOCS_URL } from "#/manifests/automation-interface";
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
  it("explains import and offers a drop zone or file picker", () => {
    render(
      <ImportAutomationModal
        isOpen
        spec={null}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
        onFile={vi.fn()}
      />,
    );

    const modal = screen.getByTestId("import-automation-modal");
    expect(modal).toHaveAttribute("data-view", "picker");
    expect(modal).toHaveTextContent(I18nKey.AUTOMATIONS$IMPORT_EXPLAIN);
    expect(modal).toHaveTextContent(I18nKey.AUTOMATIONS$IMPORT_DISABLED_NOTICE);
    expect(
      screen.getByTestId("import-automation-dropzone"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("import-automation-choose-file"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("automations-import-file")).toBeInTheDocument();
    const docsLink = screen.getByTestId("import-automation-format-docs");
    expect(docsLink).toHaveAttribute("href", AUTOMATION_FILE_FORMAT_DOCS_URL);
    expect(docsLink).toHaveTextContent(I18nKey.AUTOMATIONS$IMPORT_FORMAT_DOCS);
  });

  it("passes a dropped file to onFile", () => {
    const onFile = vi.fn();
    render(
      <ImportAutomationModal
        isOpen
        spec={null}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
        onFile={onFile}
      />,
    );

    const file = new File(['{"name":"x"}'], "automation.json", {
      type: "application/json",
    });
    fireEvent.drop(screen.getByTestId("import-automation-dropzone"), {
      dataTransfer: { files: [file] },
    });

    expect(onFile).toHaveBeenCalledWith(file);
  });

  it("previews the parsed automation before import", () => {
    const markup = renderToStaticMarkup(
      <ImportAutomationModal
        isOpen
        spec={spec}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
        onFile={vi.fn()}
      />,
    );

    expect(markup).toContain('data-testid="import-automation-modal"');
    expect(markup).toContain('data-view="preview"');
    expect(markup).toContain(spec.name);
    expect(markup).toContain("github: pull_request.opened");
    expect(markup).toContain(spec.prompt!);
    expect(markup).toContain(spec.plugins!.join(", "));
    expect(markup).toContain(I18nKey.AUTOMATIONS$IMPORT_DISABLED_NOTICE);
    expect(markup).toContain('data-testid="import-automation-confirm"');
  });

  it("does not render when closed", () => {
    const markup = renderToStaticMarkup(
      <ImportAutomationModal
        isOpen={false}
        spec={null}
        isImporting={false}
        onClose={vi.fn()}
        onImport={vi.fn()}
        onFile={vi.fn()}
      />,
    );

    expect(markup).toBe("");
  });
});
