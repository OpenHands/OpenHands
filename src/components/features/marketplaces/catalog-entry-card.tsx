import React from "react";
import { useTranslation } from "react-i18next";
import { ArrowUpCircle, TerminalSquare } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CirclePlusCheckToggle } from "#/components/shared/buttons/circle-plus-check-toggle";
import {
  SKILL_CARD_PILL_CLASS,
  SkillCardPillRow,
  type SkillCardPill,
} from "#/components/features/skills/skill-card-pill-row";
import { extensionModuleCardSurfaceClassName } from "#/utils/extension-module-card-classes";
import type { CatalogEntry } from "#/types/marketplaces";
import type { EntryAction } from "#/hooks/mutation/use-manage-marketplace-entries";
import { CatalogEntryBadge } from "./catalog-entry-badge";

interface CatalogEntryCardProps {
  entry: CatalogEntry;
  /** The marketplace this entry comes from is marked trusted. */
  marketplaceTrusted: boolean;
  isBusy?: boolean;
  onAction: (action: EntryAction, enabled?: boolean) => void;
}

/** The MCP library card's subtitle: how the server is reached. */
function subtitleFor(entry: CatalogEntry): string {
  if (entry.kind === "mcp") return (entry.transport ?? "").toUpperCase();
  const parts = [entry.marketplace];
  if (entry.version) parts.push(`v${entry.version}`);
  return parts.join(" · ");
}

/**
 * One card for every kind, shaped like the MCP library and Skills cards: icon
 * badge, title and subtitle, the circled +/check control at the top right,
 * the description, then a pill row. The control installs an entry that is not
 * installed yet and switches an installed one on and off.
 */
export function CatalogEntryCard({
  entry,
  marketplaceTrusted,
  isBusy = false,
  onAction,
}: CatalogEntryCardProps) {
  const { t } = useTranslation("openhands");
  const installed = entry.install.state !== "not_installed";
  const updateAvailable = entry.install.state === "update_available";
  const trustBlocked = entry.runs_code && !marketplaceTrusted && !installed;
  /** Templates become automations when installed; they have no on/off state. */
  const canToggle = entry.kind !== "automation";

  const pills = React.useMemo<SkillCardPill[]>(() => {
    const result: SkillCardPill[] = [];
    if (entry.runs_code) {
      result.push({
        id: "runs-code",
        node: (
          <span
            data-testid={`catalog-entry-runs-code-${entry.kind}-${entry.name}`}
            className={cn(SKILL_CARD_PILL_CLASS, "gap-1 text-warning")}
            title={t(I18nKey.MARKETPLACES$RUNS_CODE_HELP)}
          >
            <TerminalSquare className="h-3 w-3" aria-hidden />
            {t(I18nKey.MARKETPLACES$RUNS_CODE)}
          </span>
        ),
      });
    }
    if (entry.requires.length > 0) {
      result.push({
        id: "requires",
        node: (
          <span className={SKILL_CARD_PILL_CLASS}>
            {t(I18nKey.MARKETPLACES$REQUIRES, {
              items: entry.requires.join(", "),
            })}
          </span>
        ),
      });
    }
    for (const other of entry.also_in) {
      result.push({
        id: `also-in-${other}`,
        node: (
          <span
            data-testid={`catalog-entry-also-in-${entry.kind}-${entry.name}`}
            className={SKILL_CARD_PILL_CLASS}
          >
            {t(I18nKey.MARKETPLACES$ALSO_IN, { marketplace: other })}
          </span>
        ),
      });
    }
    for (const tag of entry.tags) {
      result.push({
        id: `tag-${tag}`,
        node: <span className={SKILL_CARD_PILL_CLASS}>{tag}</span>,
      });
    }
    return result;
  }, [entry, t]);

  return (
    <div
      data-testid={`catalog-entry-${entry.kind}-${entry.name}`}
      className={cn(
        "flex min-h-33 flex-col overflow-hidden p-4 text-left",
        extensionModuleCardSurfaceClassName,
      )}
    >
      <div className="flex items-start gap-3">
        <CatalogEntryBadge entry={entry} />
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <header className="flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-sm font-semibold">{entry.title}</h3>
              <p
                data-testid={`catalog-entry-kind-${entry.kind}-${entry.name}`}
                className="mt-0.5 truncate text-xs text-tertiary-alt"
                title={`${t(I18nKey.MARKETPLACES$KIND_FILTER_LABEL)}: ${entry.kind}`}
              >
                {subtitleFor(entry)}
              </p>
            </div>
            <CirclePlusCheckToggle
              testId={
                installed
                  ? `catalog-entry-toggle-${entry.kind}-${entry.name}`
                  : `catalog-entry-install-${entry.kind}-${entry.name}`
              }
              isSelected={installed && (!canToggle || entry.install.enabled)}
              isDisabled={isBusy || trustBlocked || (installed && !canToggle)}
              onToggle={(selected) => {
                if (!installed) {
                  if (selected) onAction("install");
                  return;
                }
                onAction("enable", selected);
              }}
              enableLabelKey={
                installed
                  ? I18nKey.MARKETPLACES$ENABLE
                  : I18nKey.MARKETPLACES$ACTION_INSTALL
              }
              disableLabelKey={I18nKey.MARKETPLACES$DISABLE}
              disableTooltipKey={I18nKey.MARKETPLACES$DISABLE}
            />
          </header>

          {entry.description ? (
            <p className="line-clamp-3 text-xs leading-relaxed text-tertiary-light">
              {entry.description}
            </p>
          ) : null}

          {trustBlocked ? (
            <p
              data-testid={`catalog-entry-trust-required-${entry.kind}-${entry.name}`}
              className="text-xs text-warning"
            >
              {t(I18nKey.MARKETPLACES$TRUST_REQUIRED)}
            </p>
          ) : null}

          {pills.length > 0 ? (
            <SkillCardPillRow
              pills={pills}
              testId={`catalog-entry-pills-${entry.kind}-${entry.name}`}
            />
          ) : null}

          {installed ? (
            <footer className="flex flex-wrap items-center justify-between gap-2">
              <span className="min-w-0 truncate text-xs text-tertiary-light">
                {entry.install.installed_ref
                  ? t(I18nKey.MARKETPLACES$INSTALLED_FROM, {
                      ref: entry.install.installed_ref,
                    })
                  : t(I18nKey.MARKETPLACES$ACTION_INSTALLED)}
              </span>
              <div className="flex shrink-0 gap-2">
                {updateAvailable ? (
                  <BrandButton
                    type="button"
                    variant="primary"
                    testId={`catalog-entry-update-${entry.kind}-${entry.name}`}
                    isDisabled={isBusy}
                    startContent={
                      <ArrowUpCircle className="h-3.5 w-3.5" aria-hidden />
                    }
                    onClick={() => onAction("update")}
                  >
                    {t(I18nKey.MARKETPLACES$ACTION_UPDATE)}
                  </BrandButton>
                ) : null}
                <BrandButton
                  type="button"
                  variant="secondary"
                  testId={`catalog-entry-uninstall-${entry.kind}-${entry.name}`}
                  isDisabled={isBusy}
                  onClick={() => onAction("uninstall")}
                >
                  {t(I18nKey.MARKETPLACES$ACTION_UNINSTALL)}
                </BrandButton>
              </div>
            </footer>
          ) : null}
        </div>
      </div>
    </div>
  );
}
