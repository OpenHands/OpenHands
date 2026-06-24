import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { MarketplaceRegistration } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface MarketplaceModalProps {
  isOpen: boolean;
  mode: "add" | "edit";
  scope?: "org" | "personal"; // Only for add mode
  marketplace?: MarketplaceRegistration | null;
  onClose: () => void;
  onSave: (data: {
    name: string;
    source: string;
    ref?: string;
    repo_path?: string;
    auto_load?: "all";
    scope: "org" | "personal";
  }) => void;
  onDelete?: () => void;
  isSaving?: boolean;
  isDeleting?: boolean;
  isAdminOrOwner?: boolean; // For org scope availability
}

export function MarketplaceModal({
  isOpen,
  mode,
  scope: initialScope,
  marketplace,
  onClose,
  onSave,
  onDelete,
  isSaving = false,
  isDeleting = false,
  isAdminOrOwner = false,
}: MarketplaceModalProps) {
  const { t } = useTranslation();
  const [name, setName] = React.useState(marketplace?.name || "");
  const [source, setSource] = React.useState(marketplace?.source || "");
  const [ref, setRef] = React.useState(marketplace?.ref || "");
  const [repoPath, setRepoPath] = React.useState(marketplace?.repo_path || "");
  const [autoLoad, setAutoLoad] = React.useState(
    marketplace?.auto_load === "all",
  );
  const [scope, setScope] = React.useState<"org" | "personal">(
    initialScope || "personal",
  );
  const [nameError, setNameError] = React.useState<string | null>(null);

  // Reset form when modal opens/closes or marketplace changes
  React.useEffect(() => {
    if (isOpen) {
      setName(marketplace?.name || "");
      setSource(marketplace?.source || "");
      setRef(marketplace?.ref || "");
      setRepoPath(marketplace?.repo_path || "");
      setAutoLoad(marketplace?.auto_load === "all");
      setScope(initialScope || "personal");
      setNameError(null);
    }
  }, [isOpen, marketplace, initialScope]);

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
      return;
    }
    setNameError(null);
    onSave({
      name: name.trim(),
      source: source.trim(),
      ref: ref.trim() || undefined,
      repo_path: repoPath.trim() || undefined,
      auto_load: autoLoad ? "all" : undefined,
      scope,
    });
  };

  if (!isOpen) return null;

  const canAddOrg = isAdminOrOwner;
  const isEdit = mode === "edit";

  const footer = (
    <div className="w-full flex gap-2 mt-2">
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={onClose}
        isDisabled={isSaving || isDeleting}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      {isEdit && onDelete && (
        <BrandButton
          testId="marketplace-delete-button"
          type="button"
          variant="danger"
          className="grow"
          onClick={onDelete}
          isDisabled={isDeleting}
        >
          {isDeleting ? <span>...</span> : t(I18nKey.BUTTON$DELETE)}
        </BrandButton>
      )}
      <BrandButton
        testId="marketplace-save-button"
        type="button"
        variant="primary"
        className="grow"
        onClick={handleSave}
        isDisabled={isSaving || isDeleting}
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

        {/* Scope selector - only for add mode */}
        {!isEdit && (
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
            </label>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value as "org" | "personal")}
              disabled={scope === "org" && !canAddOrg}
              className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 text-sm disabled:opacity-50"
              title={
                !canAddOrg
                  ? t(I18nKey.SETTINGS$MARKETPLACE_ORG_REQUIRE_ADMIN)
                  : undefined
              }
            >
              <option value="personal">
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_PERSONAL)}
              </option>
              <option value="org" disabled={!canAddOrg}>
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_ORG)}
              </option>
            </select>
          </div>
        )}

        {/* Name field */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_NAME)}
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setNameError(null);
            }}
            placeholder="e.g., my-skills"
            className={cn(
              "bg-tertiary border h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt",
              nameError && "border-red-500",
            )}
          />
          {nameError && (
            <Typography.Paragraph className="text-xs text-red-400">
              {nameError}
            </Typography.Paragraph>
          )}
        </div>

        {/* Source field - read-only in edit mode */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
          </label>
          <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="github:owner/repo"
            disabled={isEdit}
            readOnly={isEdit}
            className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {isEdit && (
            <Typography.Paragraph className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE_READONLY)}
            </Typography.Paragraph>
          )}
        </div>

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
            onChange={(e) => setRepoPath(e.target.value)}
            placeholder="e.g., marketplaces/internal"
            className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt"
          />
        </div>

        {/* Auto-load toggle */}
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-tertiary-alt">
            {t(I18nKey.SETTINGS$MARKETPLACE_AUTO_LOAD)}
          </label>
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
