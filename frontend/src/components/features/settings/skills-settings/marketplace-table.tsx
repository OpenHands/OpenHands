import React from "react";
import { useTranslation } from "react-i18next";
import { MarketplaceRegistration } from "#/types/settings";
import { Toggle } from "#/components/shared/toggle/toggle";
import { InfoTooltip } from "#/components/features/settings/info-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";

interface MarketplaceTableProps {
  marketplaces: MarketplaceRegistration[];
  onToggleAutoLoad: (name: string) => void;
  onEdit: (marketplace: MarketplaceRegistration) => void;
  onDelete: (marketplace: MarketplaceRegistration) => void;
  canEdit: (marketplace: MarketplaceRegistration) => boolean;
  getAutoLoadTitle: (
    scope: "instance" | "org" | "personal",
  ) => string | undefined;
  isAdminOrOwner: boolean;
}

function ScopeBadge({ scope }: { scope: "instance" | "org" | "personal" }) {
  const { t } = useTranslation();
  const label = {
    instance: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_INSTANCE),
    org: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_ORG),
    personal: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_PERSONAL),
  }[scope];

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        scope === "instance" && "bg-tertiary/60 text-tertiary-light",
        scope === "org" && "bg-blue-500/15 text-blue-300",
        scope === "personal" && "bg-success/15 text-success",
      )}
    >
      {label}
    </span>
  );
}

const HEADER_CLASS =
  "px-4 py-3 text-left text-xs font-medium text-tertiary-alt whitespace-nowrap";
const CELL_CLASS = "px-4 py-3 align-middle";

export function MarketplaceTable({
  marketplaces,
  onToggleAutoLoad,
  onEdit,
  onDelete,
  canEdit,
  getAutoLoadTitle,
  isAdminOrOwner,
}: MarketplaceTableProps) {
  const { t } = useTranslation();

  return (
    <div className="overflow-hidden rounded-xl border border-tertiary bg-base-secondary/20 table-box-shadow">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] table-fixed border-collapse text-sm">
          <colgroup>
            <col className="w-[17%]" />
            <col className="w-[21%]" />
            <col className="w-[15%]" />
            <col className="w-[15%]" />
            <col className="w-[11%]" />
            <col className="w-[10%]" />
            <col className="w-[11%]" />
          </colgroup>
          <thead>
            <tr className="border-b border-tertiary bg-base-secondary/50">
              <th className={HEADER_CLASS}>
                {t(I18nKey.SETTINGS$MARKETPLACE_NAME)}
              </th>
              <th className={HEADER_CLASS}>
                {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
              </th>
              <th className={HEADER_CLASS}>
                {t(I18nKey.SETTINGS$MARKETPLACE_REF)}
              </th>
              <th className={HEADER_CLASS}>
                {t(I18nKey.SETTINGS$MARKETPLACE_REPO_PATH)}
              </th>
              <th className={HEADER_CLASS}>
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
              </th>
              <th className={HEADER_CLASS}>
                <span className="inline-flex items-center gap-1.5">
                  {t(I18nKey.SETTINGS$AUTO_LOAD)}
                  <InfoTooltip
                    content={t(I18nKey.SETTINGS$AUTO_LOAD_TOOLTIP)}
                  />
                </span>
              </th>
              <th className={cn(HEADER_CLASS, "text-right")}>
                {t(I18nKey.SETTINGS$ACTIONS)}
              </th>
            </tr>
          </thead>
          <tbody>
            {marketplaces.map((mp) => (
              <tr
                key={mp.name}
                className="border-t border-tertiary/60 transition-colors hover:bg-base-secondary/40"
              >
                <td
                  className={cn(CELL_CLASS, "font-medium text-content-2")}
                  title={mp.name}
                >
                  <span className="block truncate">{mp.name}</span>
                </td>
                <td className={cn(CELL_CLASS, "text-tertiary-alt")}>
                  <span className="block truncate" title={mp.source}>
                    {mp.source}
                  </span>
                </td>
                <td className={cn(CELL_CLASS, "text-tertiary-alt")}>
                  <span className="block truncate" title={mp.ref || undefined}>
                    {mp.ref || "—"}
                  </span>
                </td>
                <td className={cn(CELL_CLASS, "text-tertiary-alt")}>
                  <span
                    className="block truncate"
                    title={mp.repo_path || undefined}
                  >
                    {mp.repo_path || "—"}
                  </span>
                </td>
                <td className={CELL_CLASS}>
                  <ScopeBadge scope={mp.scope ?? "personal"} />
                </td>
                <td className={CELL_CLASS}>
                  <Toggle
                    checked={!!mp.auto_load}
                    disabled={
                      mp.scope === "instance" ||
                      (mp.scope === "org" && !isAdminOrOwner)
                    }
                    onClick={
                      mp.scope !== "instance" &&
                      (mp.scope === "personal" || isAdminOrOwner)
                        ? () => onToggleAutoLoad(mp.name)
                        : undefined
                    }
                    title={getAutoLoadTitle(mp.scope ?? "personal")}
                    aria-label={`Toggle auto-load for ${mp.name}`}
                  />
                </td>
                <td className={CELL_CLASS}>
                  <div className="flex items-center justify-end gap-1">
                    <button
                      type="button"
                      onClick={() => onEdit(mp)}
                      disabled={!canEdit(mp)}
                      title={canEdit(mp) ? t(I18nKey.BUTTON$EDIT) : undefined}
                      aria-label={`${t(I18nKey.BUTTON$EDIT)} ${mp.name}`}
                      className={cn(
                        "rounded-md p-1.5 transition-colors",
                        canEdit(mp)
                          ? "text-tertiary-light hover:bg-white/10 hover:text-content-2"
                          : "cursor-not-allowed text-tertiary-alt opacity-40",
                      )}
                    >
                      <EditIcon width={16} height={16} />
                    </button>
                    <button
                      type="button"
                      onClick={() => onDelete(mp)}
                      disabled={!canEdit(mp)}
                      title={canEdit(mp) ? t(I18nKey.BUTTON$DELETE) : undefined}
                      aria-label={`${t(I18nKey.BUTTON$DELETE)} ${mp.name}`}
                      className={cn(
                        "rounded-md p-1.5 transition-colors",
                        canEdit(mp)
                          ? "text-danger hover:bg-danger/15"
                          : "cursor-not-allowed text-tertiary-alt opacity-40",
                      )}
                    >
                      <DeleteIcon width={16} height={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {marketplaces.length === 0 && (
              <tr className="border-t border-tertiary/60">
                <td
                  colSpan={7}
                  className="px-4 py-10 text-center text-sm text-tertiary-alt"
                >
                  {t(I18nKey.SETTINGS$MARKETPLACE_ADD_FIRST)}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
