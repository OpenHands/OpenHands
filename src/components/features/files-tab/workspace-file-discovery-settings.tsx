import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useWorkspaceFileDiscovery } from "#/hooks/query/use-workspace-file-discovery";
import {
  DEFAULT_FILE_DISCOVERY,
  isValidFileLimit,
  type WorkspaceFileDiscovery,
} from "#/utils/workspace-file-discovery";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { BrandButton } from "#/components/features/settings/brand-button";

function DiscoveryForm({
  options,
  save,
  isSaving,
  onClose,
}: {
  options: WorkspaceFileDiscovery;
  save: (options: WorkspaceFileDiscovery) => Promise<unknown>;
  isSaving: boolean;
  onClose: () => void;
}) {
  const { t } = useTranslation("openhands");
  // An unsaved form draft is discarded on cancel or workspace navigation.
  const [patterns, setPatterns] = useState(options.excludedPatterns.join("\n"));
  const [limit, setLimit] = useState(String(options.maxFiles));
  const [includeSymlinks, setIncludeSymlinks] = useState(
    options.includeSymlinks,
  );
  const [saveFailed, setSaveFailed] = useState(false);
  const maxFiles = Number(limit);
  const valid =
    limit.trim() !== "" &&
    isValidFileLimit(maxFiles) &&
    !patterns.includes("\0");
  const submit = async (value: WorkspaceFileDiscovery) => {
    setSaveFailed(false);
    try {
      await save(value);
      onClose();
    } catch {
      setSaveFailed(true);
    }
  };

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.FILES$DISCOVERY_SETTINGS)}
      closeOnEscape={!isSaving}
      closeOnBackdropClick={!isSaving}
    >
      <ModalBody
        width="md"
        className="items-stretch max-h-[90vh] overflow-y-auto"
      >
        <h2 className="text-lg font-semibold">
          {t(I18nKey.FILES$DISCOVERY_SETTINGS)}
        </h2>
        <p className="text-sm text-muted">{t(I18nKey.FILES$DISCOVERY_SCOPE)}</p>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            if (valid && !isSaving)
              void submit({
                maxFiles,
                includeSymlinks,
                excludedPatterns: patterns
                  .split(/\r?\n/)
                  .map((pattern) => pattern.trim())
                  .filter(Boolean),
              });
          }}
        >
          <label className="flex flex-col gap-2 text-sm">
            {t(I18nKey.FILES$DISCOVERY_EXCLUSIONS)}
            <textarea
              value={patterns}
              onChange={(event) => setPatterns(event.target.value)}
              rows={5}
              disabled={isSaving}
              className="rounded-md border border-border bg-base px-3 py-2 font-mono"
            />
            <span className="text-xs text-muted">
              {t(I18nKey.FILES$DISCOVERY_PATTERN_HELP)}
            </span>
          </label>
          <label className="flex flex-col gap-2 text-sm">
            {t(I18nKey.FILES$DISCOVERY_LIMIT)}
            <input
              type="number"
              min={0}
              step={1}
              max={Number.MAX_SAFE_INTEGER - 1}
              required
              value={limit}
              disabled={isSaving}
              onChange={(event) => setLimit(event.target.value)}
              className="rounded-md border border-border bg-base px-3 py-2"
            />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={includeSymlinks}
              disabled={isSaving}
              onChange={(event) => setIncludeSymlinks(event.target.checked)}
            />
            {t(I18nKey.FILES$DISCOVERY_SYMLINKS)}
          </label>
          {saveFailed && (
            <p role="alert" className="text-sm text-danger">
              {t(I18nKey.FILES$DISCOVERY_SAVE_ERROR)}
            </p>
          )}
          <div className="flex flex-wrap justify-end gap-2">
            <BrandButton
              type="button"
              variant="secondary"
              isDisabled={isSaving}
              onClick={() => {
                setPatterns(DEFAULT_FILE_DISCOVERY.excludedPatterns.join("\n"));
                setLimit(String(DEFAULT_FILE_DISCOVERY.maxFiles));
                setIncludeSymlinks(DEFAULT_FILE_DISCOVERY.includeSymlinks);
              }}
            >
              {t(I18nKey.FILES$DISCOVERY_RESET)}
            </BrandButton>
            <BrandButton
              type="button"
              variant="secondary"
              isDisabled={isSaving}
              onClick={onClose}
            >
              {t(I18nKey.BUTTON$CANCEL)}
            </BrandButton>
            <BrandButton
              type="submit"
              variant="primary"
              isDisabled={!valid || isSaving}
            >
              {t(isSaving ? I18nKey.SETTINGS$SAVING : I18nKey.BUTTON$SAVE)}
            </BrandButton>
          </div>
        </form>
      </ModalBody>
    </ModalBackdrop>
  );
}

function DiscoveryControl({
  discovery,
}: {
  discovery: ReturnType<typeof useWorkspaceFileDiscovery>;
}) {
  const { t } = useTranslation("openhands");
  const [isOpen, setIsOpen] = useState(false);
  if (!discovery.canConfigure) return null;
  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="px-2 py-1 text-xs text-muted hover:text-contrast"
        data-testid="file-discovery-settings"
      >
        {t(I18nKey.FILES$DISCOVERY_SETTINGS)}
      </button>
      {isOpen && (
        <DiscoveryForm
          options={discovery.options}
          save={discovery.save}
          isSaving={discovery.isSaving}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}

// @spec WFD-002 — Workspace-scoped server persistence
export function WorkspaceFileDiscoverySettings({
  workingDir,
}: {
  workingDir?: string;
}) {
  const discovery = useWorkspaceFileDiscovery(workingDir);
  return <DiscoveryControl key={discovery.scopeKey} discovery={discovery} />;
}
