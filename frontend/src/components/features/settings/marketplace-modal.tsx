import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { InfoTooltip } from "#/components/features/settings/info-tooltip";
import { MarketplaceRegistration } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

/**
 * Best-effort marketplace name from a repository source, mirroring the backend
 * `_derive_name_from_source` (marketplace_composition.py): strip a `provider:`
 * prefix, take the last path segment, drop a trailing `.git`. The result is then
 * sanitized to the valid name pattern (`^[a-zA-Z][a-zA-Z0-9_-]*$`) so the derived
 * default is usually accepted as-is; on-save validation still guards edge cases.
 */
export function deriveMarketplaceName(source: string): string {
  let s = (source || "").trim();
  if (!s) return "";
  if (s.includes(":")) s = s.slice(s.indexOf(":") + 1);
  if (s.includes("/")) s = s.replace(/\/+$/, "").split("/").pop() ?? "";
  s = s.replace(/\.git$/i, "");
  return s
    .replace(/[^a-zA-Z0-9_-]+/g, "-")
    .replace(/^[^a-zA-Z]+/, "")
    .replace(/-+$/g, "");
}

interface MarketplaceModalProps {
  isOpen: boolean;
  mode: "add" | "edit";
  marketplace?: MarketplaceRegistration | null;
  /**
   * When true (admin adding in an org context), the modal shows an org/personal
   * scope selector. Otherwise the marketplace is added as personal.
   */
  allowScopeSelection?: boolean;
  onClose: () => void;
  onSave: (data: {
    name: string;
    source: string;
    ref?: string;
    repo_path?: string;
    auto_load?: boolean;
    scope?: "org" | "personal";
  }) => void;
  isSaving?: boolean;
}

export function MarketplaceModal({
  isOpen,
  mode,
  marketplace,
  allowScopeSelection = false,
  onClose,
  onSave,
  isSaving = false,
}: MarketplaceModalProps) {
  const { t } = useTranslation();
  const [name, setName] = React.useState(marketplace?.name || "");
  const [source, setSource] = React.useState(marketplace?.source || "");
  const [ref, setRef] = React.useState(marketplace?.ref || "");
  const [repoPath, setRepoPath] = React.useState(marketplace?.repo_path || "");
  const [autoLoad, setAutoLoad] = React.useState(!!marketplace?.auto_load);
  const [scope, setScope] = React.useState<"org" | "personal">("personal");
  // Tracks whether the user has hand-edited Name, so auto-derivation from the
  // repository stops clobbering an intentional name.
  const [nameEdited, setNameEdited] = React.useState(false);
  const [nameError, setNameError] = React.useState<string | null>(null);
  const [sourceError, setSourceError] = React.useState<string | null>(null);
  const [repoPathError, setRepoPathError] = React.useState<string | null>(null);

  // Reset form when modal opens/closes or marketplace changes
  React.useEffect(() => {
    if (isOpen) {
      setName(marketplace?.name || "");
      setSource(marketplace?.source || "");
      setRef(marketplace?.ref || "");
      setRepoPath(marketplace?.repo_path || "");
      setAutoLoad(!!marketplace?.auto_load);
      // Default new marketplaces to personal; admins can opt into org.
      setScope(marketplace?.scope === "org" ? "org" : "personal");
      // An existing (edit) name counts as user-owned; a fresh add starts clean.
      setNameEdited(Boolean(marketplace?.name));
      setNameError(null);
      setSourceError(null);
      setRepoPathError(null);
    }
  }, [isOpen, marketplace]);

  // Scope selection only applies when adding and the caller allows it.
  const showScopeSelector = mode === "add" && allowScopeSelection;

  const handleSave = () => {
    // Validate name on click
    if (!name.trim()) {
      setNameError(t(I18nKey.SETTINGS$MARKETPLACE_NAME_REQUIRED));
      return;
    }
    if (!/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(name.trim())) {
      setNameError(t(I18nKey.SETTINGS$MARKETPLACE_NAME_INVALID));
      return;
    }
    if (!source.trim()) {
      setSourceError(t(I18nKey.SETTINGS$MARKETPLACE_SOURCE_REQUIRED));
      return;
    }
    // Repository Path is an optional subdirectory *within* the repository
    // (e.g. "marketplaces/internal"), not a URL or absolute path. Catch the
    // common mistake of pasting the repository URL here before it reaches the
    // backend, where it would fail with a confusing "Repo path not found".
    const trimmedRepoPath = repoPath.trim();
    if (
      trimmedRepoPath &&
      (trimmedRepoPath.includes("://") ||
        trimmedRepoPath.startsWith("/") ||
        trimmedRepoPath.startsWith("\\") ||
        trimmedRepoPath.includes(".."))
    ) {
      setRepoPathError(t(I18nKey.SETTINGS$MARKETPLACE_REPO_PATH_INVALID));
      return;
    }
    setNameError(null);
    setRepoPathError(null);

    onSave({
      name: name.trim(),
      source: source.trim(),
      ref: ref.trim() || undefined,
      repo_path: trimmedRepoPath || undefined,
      auto_load: autoLoad,
      scope: showScopeSelector ? scope : undefined,
    });
  };

  if (!isOpen) return null;

  const isEdit = mode === "edit";

  const footer = (
    <div className="w-full flex gap-2 mt-2">
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={onClose}
        isDisabled={isSaving}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="marketplace-save-button"
        type="button"
        variant="primary"
        className="grow"
        onClick={handleSave}
        isDisabled={isSaving}
      >
        {isSaving ? <span>...</span> : t(I18nKey.BUTTON$SAVE)}
      </BrandButton>
    </div>
  );

  return (
    <ModalBackdrop>
      <div
        className="bg-base p-6 rounded-xl flex flex-col gap-4 border border-tertiary"
        style={{ width: "500px" }}
      >
        <h3 className="text-xl font-bold">
          {isEdit
            ? t(I18nKey.SETTINGS$MARKETPLACE_EDIT_TITLE)
            : t(I18nKey.SETTINGS$MARKETPLACE_ADD_TITLE)}
        </h3>

        {/* Repository (source) field - required, no default. Shown first so the
            Name below can be derived from it. Read-only in edit mode. */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
          </label>
          <input
            type="text"
            value={source}
            onChange={(e) => {
              const nextSource = e.target.value;
              setSource(nextSource);
              setSourceError(null);
              // Pre-fill Name from the repository until the user edits Name.
              if (!isEdit && !nameEdited) {
                setName(deriveMarketplaceName(nextSource));
                setNameError(null);
              }
            }}
            placeholder="github:owner/repo"
            disabled={isEdit}
            readOnly={isEdit}
            className={cn(
              "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt disabled:opacity-50 disabled:cursor-not-allowed",
              sourceError && "border-red-500",
            )}
          />
          {sourceError && (
            <Typography.Paragraph className="text-xs text-red-400">
              {sourceError}
            </Typography.Paragraph>
          )}
          {!sourceError && isEdit && (
            <Typography.Paragraph className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE_READONLY)}
            </Typography.Paragraph>
          )}
        </div>

        {/* Name field - defaults to a value derived from the repository above. */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_NAME)}
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setNameEdited(true);
              setNameError(null);
            }}
            placeholder="e.g., my-skills"
            className={cn(
              "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt",
              nameError && "border-red-500",
            )}
          />
          {nameError && (
            <Typography.Paragraph className="text-xs text-red-400">
              {nameError}
            </Typography.Paragraph>
          )}
        </div>

        {/* Scope selector - only when an admin adds in an org context */}
        {showScopeSelector && (
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
            </label>
            <select
              data-testid="marketplace-scope-select"
              value={scope}
              onChange={(e) => setScope(e.target.value as "org" | "personal")}
              className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2"
            >
              <option value="personal">
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_PERSONAL)}
              </option>
              <option value="org">
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_ORG)}
              </option>
            </select>
          </div>
        )}

        {/* Ref field (optional) */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_REF)}
            <span className="text-tertiary-alt font-normal ml-1">
              ({t(I18nKey.SETTINGS$OPTIONAL)})
            </span>
          </label>
          <input
            type="text"
            value={ref}
            onChange={(e) => setRef(e.target.value)}
            placeholder="e.g., main, develop, v1.0.0"
            className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt"
          />
        </div>

        {/* Repo path field (optional) */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_REPO_PATH)}
            <span className="text-tertiary-alt font-normal ml-1">
              ({t(I18nKey.SETTINGS$OPTIONAL)})
            </span>
          </label>
          <input
            type="text"
            value={repoPath}
            onChange={(e) => {
              setRepoPath(e.target.value);
              setRepoPathError(null);
            }}
            placeholder="e.g., marketplaces/internal"
            className={cn(
              "bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt",
              repoPathError && "border-red-500",
            )}
          />
          {repoPathError && (
            <Typography.Paragraph className="text-xs text-red-400">
              {repoPathError}
            </Typography.Paragraph>
          )}
        </div>

        {/* Auto-load toggle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <label className="text-sm font-medium text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_AUTO_LOAD)}
            </label>
            <InfoTooltip content={t(I18nKey.SETTINGS$AUTO_LOAD_TOOLTIP)} />
          </div>
          <button
            type="button"
            onClick={() => setAutoLoad(!autoLoad)}
            aria-label={t(I18nKey.SETTINGS$MARKETPLACE_AUTO_LOAD)}
            className="cursor-pointer"
          >
            <div
              className={cn(
                "w-12 h-6 rounded-xl flex items-center p-1.5",
                autoLoad && "justify-end bg-white",
                !autoLoad && "justify-start bg-base-secondary",
              )}
            >
              <div
                className={cn(
                  "w-3 h-3 rounded-xl",
                  autoLoad ? "bg-[#0D0F11]" : "bg-tertiary-light",
                )}
              />
            </div>
          </button>
        </div>

        {footer}
      </div>
    </ModalBackdrop>
  );
}
