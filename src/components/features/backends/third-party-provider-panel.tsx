import React from "react";
import { useTranslation } from "react-i18next";
import { ArrowLeft, Check, ChevronRight, Plus } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import DigitalOceanIcon from "#/icons/digitalocean.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { cn } from "#/utils/utils";
import {
  DIGITALOCEAN_MARKETPLACE_URL,
  DIGITALOCEAN_PROVIDER_ID,
  MOCK_DIGITALOCEAN_DROPLETS,
  THIRD_PARTY_BACKEND_PROVIDERS,
  type MockDigitalOceanDroplet,
  type ThirdPartyBackendProvider,
  type ThirdPartyBackendProviderId,
} from "./third-party-backend-providers";

function ProviderLogo({
  providerId,
  className,
}: {
  providerId: ThirdPartyBackendProviderId;
  className?: string;
}) {
  if (providerId === DIGITALOCEAN_PROVIDER_ID) {
    return <DigitalOceanIcon className={className} aria-hidden />;
  }
  return null;
}

function ProviderCatalogCard({
  provider,
  onSelect,
}: {
  provider: ThirdPartyBackendProvider;
  onSelect: (id: ThirdPartyBackendProviderId) => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      data-testid={`add-backend-provider-${provider.id}`}
      onClick={() => onSelect(provider.id)}
      className={cn(
        "flex w-full items-center gap-3 rounded-xl border border-[var(--oh-border)] bg-base-secondary px-4 py-3 text-left",
        "transition-colors hover:bg-[var(--oh-surface-raised)]",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
      )}
    >
      <span
        className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--oh-surface-raised)] text-[#0080FF]"
        aria-hidden
      >
        <ProviderLogo providerId={provider.id} className="size-8" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-white">
          {provider.name}
        </span>
        <span className="mt-0.5 block text-xs leading-tight text-[var(--oh-muted)]">
          {t(provider.descriptionKey)}
        </span>
      </span>
      <ChevronRight
        className="size-4 shrink-0 text-[var(--oh-muted)]"
        aria-hidden
      />
    </button>
  );
}

export function DigitalOceanTokenStep({
  token,
  onTokenChange,
  onContinue,
  onBack,
}: {
  token: string;
  onTokenChange: (value: string) => void;
  onContinue: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("openhands");
  const canContinue = token.trim().length > 0;

  return (
    <div
      data-testid="add-backend-provider-digitalocean-token"
      className="flex flex-col gap-4"
    >
      <ProviderStepHeader
        providerId={DIGITALOCEAN_PROVIDER_ID}
        title={
          THIRD_PARTY_BACKEND_PROVIDERS.find(
            (provider) => provider.id === DIGITALOCEAN_PROVIDER_ID,
          )?.name ?? DIGITALOCEAN_PROVIDER_ID
        }
        onBack={onBack}
        backTestId="add-backend-providers-back"
      />

      <DigitalOceanCreateDropletCard />

      <div className="flex items-center gap-3">
        <span className="h-px flex-1 bg-[var(--oh-border)]" aria-hidden />
        <p className="shrink-0 text-xs text-[var(--oh-muted)]">
          {t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_CONNECT_EXISTING)}
        </p>
        <span className="h-px flex-1 bg-[var(--oh-border)]" aria-hidden />
      </div>

      <SettingsInput
        testId="add-backend-provider-digitalocean-token-input"
        name="add-backend-provider-digitalocean-token"
        type="password"
        label={t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_TOKEN_LABEL)}
        hint={t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_TOKEN_HINT)}
        value={token}
        onChange={onTokenChange}
        // eslint-disable-next-line i18next/no-literal-string -- example token shape, not user-facing copy
        placeholder="dop_v1_••••••••••"
        className="w-full"
      />

      <BrandButton
        type="button"
        variant="secondary"
        isDisabled={!canContinue}
        testId="add-backend-provider-digitalocean-continue"
        className="w-full text-center"
        onClick={onContinue}
      >
        {t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_CONTINUE)}
      </BrandButton>
    </div>
  );
}

function DigitalOceanDropletCard({
  droplet,
  selected,
  onToggle,
}: {
  droplet: MockDigitalOceanDroplet;
  selected: boolean;
  onToggle: (id: string) => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <button
      type="button"
      role="checkbox"
      data-testid={`add-backend-provider-digitalocean-droplet-${droplet.id}`}
      aria-checked={selected}
      onClick={() => onToggle(droplet.id)}
      className={cn(
        "flex w-full items-start gap-3 rounded-xl border border-[var(--oh-border)] px-4 py-3 text-left transition-colors",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
        "hover:bg-[var(--oh-surface-raised)]",
        selected && "bg-[var(--oh-surface-raised)]",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded border",
          selected
            ? "border-white bg-white text-black"
            : "border-[var(--oh-border)]",
        )}
      >
        {selected ? <Check className="size-3" strokeWidth={3} /> : null}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium text-white">
          {droplet.name}
        </span>
        <span className="mt-0.5 block text-xs text-[var(--oh-muted)]">
          {droplet.region}
          {}
          {" · "}
          {droplet.ipv4}
        </span>
      </span>
      <span
        className={cn(
          "mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-[11px] leading-4",
          droplet.status === "active"
            ? "bg-emerald-500/15 text-emerald-300"
            : "bg-[var(--oh-surface-raised)] text-[var(--oh-muted)]",
        )}
      >
        {droplet.status === "active"
          ? t(I18nKey.BACKEND$PROVIDER_DROPLET_STATUS_ACTIVE)
          : t(I18nKey.BACKEND$PROVIDER_DROPLET_STATUS_OFF)}
      </span>
    </button>
  );
}

export function DigitalOceanDropletsStep({
  selectedDropletIds,
  onToggleDroplet,
  onBack,
}: {
  selectedDropletIds: readonly string[];
  onToggleDroplet: (id: string) => void;
  onBack: () => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="add-backend-provider-digitalocean-droplets"
      className="flex flex-col gap-4"
    >
      <ProviderStepHeader
        providerId={DIGITALOCEAN_PROVIDER_ID}
        title={t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_DROPLETS_TITLE)}
        onBack={onBack}
        backTestId="add-backend-provider-digitalocean-droplets-back"
        action={
          <DigitalOceanCreateDropletButton testId="add-backend-provider-digitalocean-create-again" />
        }
      />

      <div
        className="flex flex-col gap-2"
        role="group"
        aria-label={t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_DROPLETS_TITLE)}
      >
        {MOCK_DIGITALOCEAN_DROPLETS.map((droplet) => (
          <DigitalOceanDropletCard
            key={droplet.id}
            droplet={droplet}
            selected={selectedDropletIds.includes(droplet.id)}
            onToggle={onToggleDroplet}
          />
        ))}
      </div>

      <BrandButton
        type="button"
        variant="secondary"
        isDisabled={selectedDropletIds.length === 0}
        testId="add-backend-provider-digitalocean-submit"
        className="w-full text-center"
      >
        {t(I18nKey.BACKEND$CONNECT)}
      </BrandButton>
    </div>
  );
}

function DigitalOceanCreateDropletCard() {
  const { t } = useTranslation("openhands");

  return (
    <a
      href={DIGITALOCEAN_MARKETPLACE_URL}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="add-backend-provider-digitalocean-create"
      className={cn(
        "flex w-full items-center gap-3 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-4 py-3 text-left",
        "transition-colors hover:border-white",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
      )}
    >
      <span
        className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-base-secondary text-white"
        aria-hidden
      >
        <Plus className="size-5" />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-white">
          {t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_CREATE_TITLE)}
        </span>
        <span className="mt-0.5 block text-xs leading-tight text-[var(--oh-muted)]">
          {t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_CREATE_DESCRIPTION)}
        </span>
      </span>
    </a>
  );
}

/**
 * Compact header action for the droplet list. Marketplace 1-Click covers
 * account creation and droplet create — DigitalOcean signs the user in (or up).
 */
function DigitalOceanCreateDropletButton({ testId }: { testId: string }) {
  const { t } = useTranslation("openhands");

  return (
    <a
      href={DIGITALOCEAN_MARKETPLACE_URL}
      target="_blank"
      rel="noopener noreferrer"
      data-testid={testId}
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-md border border-[var(--oh-border)] bg-base-secondary px-3 py-1.5 text-sm text-white",
        "transition-colors hover:bg-[var(--oh-surface-raised)]",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
      )}
    >
      <Plus className="size-3.5" aria-hidden />
      <span>{t(I18nKey.BACKEND$PROVIDER_DIGITALOCEAN_CREATE_TITLE)}</span>
    </a>
  );
}

function ProviderStepHeader({
  providerId,
  title,
  onBack,
  backTestId,
  action,
}: {
  providerId: ThirdPartyBackendProviderId;
  title: string;
  onBack: () => void;
  backTestId: string;
  action?: React.ReactNode;
}) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        data-testid={backTestId}
        onClick={onBack}
        className={cn(
          "inline-flex w-fit items-center gap-1.5 rounded-md py-1 text-sm text-[var(--oh-muted)]",
          "transition-colors hover:text-white",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
        )}
      >
        <ArrowLeft className="size-4" aria-hidden />
        <span>{t(I18nKey.BUTTON$BACK)}</span>
      </button>
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span
            className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[var(--oh-surface-raised)] text-[#0080FF]"
            aria-hidden
          >
            <ProviderLogo providerId={providerId} className="size-8" />
          </span>
          <h2 className="truncate text-lg font-medium text-[var(--oh-modal-title-foreground)]">
            {title}
          </h2>
        </div>
        {action}
      </div>
    </div>
  );
}

/**
 * Catalog of third-party providers. Selecting a card is handled by the parent
 * so the DigitalOcean steps can slide over the whole add-backend modal.
 */
export function ThirdPartyProviderPanel({
  onSelectProvider,
}: {
  onSelectProvider: (id: ThirdPartyBackendProviderId) => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <div data-testid="add-backend-providers-panel" className="w-full">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          {THIRD_PARTY_BACKEND_PROVIDERS.map((provider) => (
            <ProviderCatalogCard
              key={provider.id}
              provider={provider}
              onSelect={onSelectProvider}
            />
          ))}
        </div>
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.BACKEND$PROVIDERS_COMING_SOON)}
        </p>
      </div>
    </div>
  );
}
