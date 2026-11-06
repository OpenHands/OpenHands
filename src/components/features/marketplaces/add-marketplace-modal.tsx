import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { useAddMarketplace } from "#/hooks/mutation/use-manage-marketplaces";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import type { MarketplaceSource } from "#/types/marketplaces";

interface AddMarketplaceModalProps {
  onClose: () => void;
  onAdded?: (marketplace: MarketplaceSource) => void;
}

export function AddMarketplaceModal({
  onClose,
  onAdded,
}: AddMarketplaceModalProps) {
  const { t } = useTranslation("openhands");
  const addMarketplace = useAddMarketplace();

  const [source, setSource] = React.useState("");
  const [name, setName] = React.useState("");
  const [ref, setRef] = React.useState("");
  const [tokenSecret, setTokenSecret] = React.useState("");
  const [trusted, setTrusted] = React.useState(false);

  const trimmedSource = source.trim();
  const canSubmit = trimmedSource.length > 0 && !addMarketplace.isPending;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    addMarketplace.mutate(
      {
        source: trimmedSource,
        name: name.trim() || undefined,
        ref: ref.trim() || null,
        trusted,
        token_secret: tokenSecret.trim() || null,
      },
      {
        onSuccess: (marketplace) => {
          onAdded?.(marketplace);
          onClose();
        },
      },
    );
  };

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.MARKETPLACES$ADD_MODAL_TITLE)}
    >
      <form
        onSubmit={handleSubmit}
        data-testid="add-marketplace-modal"
        className="relative flex w-130 extension-dialog-viewport flex-col rounded-xl border border-border bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="add-marketplace-modal-close"
        />

        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.MARKETPLACES$ADD_MODAL_TITLE)}
          </h2>
          <p className="mt-4 text-sm text-tertiary-light">
            {t(I18nKey.MARKETPLACES$ADD_MODAL_INTRO)}
          </p>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
          <SettingsInput
            testId="add-marketplace-source-input"
            label={t(I18nKey.MARKETPLACES$SOURCE_LABEL)}
            type="text"
            value={source}
            onChange={setSource}
            placeholder={t(I18nKey.MARKETPLACES$SOURCE_PLACEHOLDER)}
            showRequiredTag
          />
          <SettingsInput
            testId="add-marketplace-name-input"
            label={t(I18nKey.MARKETPLACES$NAME_LABEL)}
            type="text"
            value={name}
            onChange={setName}
            placeholder={t(I18nKey.MARKETPLACES$NAME_PLACEHOLDER)}
            showOptionalTag
          />
          <SettingsInput
            testId="add-marketplace-ref-input"
            label={t(I18nKey.MARKETPLACES$REF_LABEL)}
            type="text"
            value={ref}
            onChange={setRef}
            showOptionalTag
          />
          <SettingsInput
            testId="add-marketplace-token-input"
            label={t(I18nKey.MARKETPLACES$TOKEN_LABEL)}
            type="text"
            value={tokenSecret}
            onChange={setTokenSecret}
            placeholder={t(I18nKey.MARKETPLACES$TOKEN_PLACEHOLDER)}
            showOptionalTag
          />
          <div className="flex items-start justify-between gap-4 rounded-lg border border-border p-3">
            <div className="min-w-0">
              <p className="text-sm text-contrast">
                {t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}
              </p>
              <p className="mt-1 text-xs text-tertiary-light">
                {t(I18nKey.MARKETPLACES$TRUSTED_HELP)}
              </p>
            </div>
            <ToggleSwitch
              enabled={trusted}
              label={t(I18nKey.MARKETPLACES$TRUSTED_LABEL)}
              onToggle={() => setTrusted((value) => !value)}
            />
          </div>
        </div>

        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="add-marketplace-modal-dismiss"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            testId="add-marketplace-submit"
            isDisabled={!canSubmit}
          >
            {t(
              addMarketplace.isPending
                ? I18nKey.MARKETPLACES$ADDING
                : I18nKey.MARKETPLACES$ADD_BUTTON,
            )}
          </BrandButton>
        </footer>
      </form>
    </ModalBackdrop>
  );
}
