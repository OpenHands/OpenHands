import React from "react";
import { Editor, type Monaco } from "@monaco-editor/react";
import type { editor as editor_t } from "monaco-editor";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useSaveWorkspaceFile } from "#/hooks/mutation/use-save-workspace-file";
import { getLanguageFromPath } from "#/utils/get-language-from-path";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

interface EditableSourceViewProps {
  path: string;
  text: string;
}

const EDITOR_OPTIONS: editor_t.IEditorOptions = {
  readOnly: false,
  renderValidationDecorations: "off",
  scrollBeyondLastLine: false,
  minimap: { enabled: false },
  automaticLayout: true,
  wordWrap: "off",
  fontSize: 12,
  lineHeight: 20,
  padding: { top: 12, bottom: 12 },
  scrollbar: { alwaysConsumeMouseWheel: false },
};

const beforeMount = (monaco: Monaco) => {
  monaco.editor.defineTheme("files-tab-editor", {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#00000000",
    },
  });
};

/**
 * Editable Monaco view for workspace text files, with Save / Discard and
 * Ctrl/Cmd+S. Replaces the read-only Prism highlighter for source editing.
 */
export function EditableSourceView({ path, text }: EditableSourceViewProps) {
  const { t } = useTranslation("openhands");
  const saveFile = useSaveWorkspaceFile();
  const [draft, setDraft] = React.useState(text);
  const [baseline, setBaseline] = React.useState(text);
  const editorRef = React.useRef<editor_t.IStandaloneCodeEditor | null>(null);

  // Reset local draft when the selected file changes or the server copy is
  // refreshed (agent edit / save / refresh). Avoid clobbering in-progress
  // typing when `text` is identical to our baseline.
  React.useEffect(() => {
    setDraft(text);
    setBaseline(text);
  }, [path, text]);

  const isDirty = draft !== baseline;
  const canSave = isDirty && !saveFile.isPending;

  const handleSave = React.useCallback(async () => {
    if (!canSave) return;
    try {
      await saveFile.mutateAsync({ relativePath: path, content: draft });
      setBaseline(draft);
      displaySuccessToast(t(I18nKey.FILES$SAVE_SUCCESS));
    } catch (err) {
      displayErrorToast(
        err instanceof Error ? err.message : t(I18nKey.FILES$SAVE_ERROR),
      );
    }
  }, [canSave, draft, path, saveFile, t]);

  const handleDiscard = React.useCallback(() => {
    setDraft(baseline);
    editorRef.current?.setValue(baseline);
  }, [baseline]);

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void handleSave();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [handleSave]);

  return (
    <div
      className="flex h-full min-h-0 w-full flex-col bg-[var(--oh-surface)]"
      data-testid="editable-source-view"
      data-dirty={isDirty ? "true" : "false"}
    >
      <div className="flex items-center justify-end gap-2 border-b border-[var(--oh-border)] px-3 py-1.5">
        {isDirty && (
          <span
            data-testid="editable-source-dirty"
            className="mr-auto text-xs text-[var(--oh-muted)]"
          >
            {t(I18nKey.FILES$UNSAVED_CHANGES)}
          </span>
        )}
        <BrandButton
          testId="editable-source-discard"
          type="button"
          variant="secondary"
          isDisabled={!isDirty || saveFile.isPending}
          onClick={handleDiscard}
        >
          {t(I18nKey.FILES$DISCARD)}
        </BrandButton>
        <BrandButton
          testId="editable-source-save"
          type="button"
          variant="primary"
          isDisabled={!canSave}
          aria-busy={saveFile.isPending}
          onClick={() => {
            void handleSave();
          }}
        >
          {t(I18nKey.FILES$SAVE)}
        </BrandButton>
      </div>
      <div className="min-h-0 flex-1">
        <Editor
          path={path}
          language={getLanguageFromPath(path)}
          theme="files-tab-editor"
          value={draft}
          options={EDITOR_OPTIONS}
          beforeMount={beforeMount}
          onMount={(editor) => {
            editorRef.current = editor;
          }}
          onChange={(value) => {
            setDraft(value ?? "");
          }}
          loading={
            <div className="flex h-full items-center justify-center text-sm text-[var(--oh-muted)]">
              {t(I18nKey.FILES$LOADING_FILES)}
            </div>
          }
        />
      </div>
    </div>
  );
}
