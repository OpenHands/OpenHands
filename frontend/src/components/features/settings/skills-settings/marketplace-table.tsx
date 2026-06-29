import React from "react";
import { useTranslation } from "react-i18next";
import { MarketplaceWithScope } from "#/types/settings";
import { Toggle } from "#/components/shared/toggle/toggle";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";

interface MarketplaceTableProps {
  marketplaces: MarketplaceWithScope[];
  onToggleAutoLoad: (source: string) => void;
  onEdit: (marketplace: MarketplaceWithScope) => void;
  onDelete: (marketplace: MarketplaceWithScope) => void;
  canEdit: (marketplace: MarketplaceWithScope) => boolean;
  getAutoLoadTitle: (scope: "instance" | "org" | "personal") => string | undefined;
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
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium",
        scope === "instance" && "bg-tertiary text-tertiary-alt",
        scope === "org" && "bg-blue-900/30 text-blue-400",
        scope === "personal" && "bg-green-900/30 text-green-400",
      )}
    >
      {label}
    </span>
  );
}

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
    <div className="border border-tertiary rounded-md overflow-hidden">
      <table className="w-full">
        <thead className="bg-base-secondary">
          <tr className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr_auto] gap-4 items-center">
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_NAME)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_REF)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_REPO_PATH)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$AUTO_LOAD)}
            </th>
            <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
              {t(I18nKey.SETTINGS$ACTIONS)}
            </th>
          </tr>
        </thead>
        <tbody>
          {marketplaces.map((mp) => (
            <tr
              key={mp.source}
              className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr_auto] gap-4 items-center border-t border-tertiary"
            >
              <td className="p-3 text-sm text-content-2 truncate min-w-0">
                {mp.name}
              </td>
              <td className="p-3 text-sm text-tertiary-alt truncate">
                {mp.source}
              </td>
              <td className="p-3 text-sm text-tertiary-alt truncate">
                {mp.ref || "-"}
              </td>
              <td className="p-3 text-sm text-tertiary-alt truncate">
                {mp.repo_path || "-"}
              </td>
              <td className="p-3">
                <ScopeBadge scope={mp.scope} />
              </td>
              <td className="p-3">
                <Toggle
                  checked={!!mp.auto_load}
                  disabled={
                    mp.scope === "instance" ||
                    (mp.scope === "org" && !isAdminOrOwner)
                  }
                  onClick={
                    mp.scope !== "instance" &&
                    (mp.scope === "personal" || isAdminOrOwner)
                      ? () => onToggleAutoLoad(mp.source)
                      : undefined
                  }
                  title={getAutoLoadTitle(mp.scope)}
                  aria-label={`Toggle auto-load for ${mp.source}`}
                />
              </td>
              <td className="p-3 flex gap-2 justify-center">
                <button
                  type="button"
                  onClick={() => onEdit(mp)}
                  disabled={!canEdit(mp)}
                  title={canEdit(mp) ? t(I18nKey.BUTTON$EDIT) : undefined}
                  className={cn(
                    "p-1.5 rounded-sm",
                    canEdit(mp)
                      ? "text-content-2 hover:bg-white/20"
                      : "text-tertiary-alt cursor-not-allowed opacity-50",
                  )}
                >
                  <EditIcon width={16} height={16} />
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(mp)}
                  disabled={!canEdit(mp)}
                  title={canEdit(mp) ? t(I18nKey.BUTTON$DELETE) : undefined}
                  className={cn(
                    "p-1.5 rounded-sm",
                    canEdit(mp)
                      ? "text-red-400 hover:bg-red-900/20"
                      : "text-tertiary-alt cursor-not-allowed opacity-50",
                  )}
                >
                  <DeleteIcon width={16} height={16} />
                </button>
              </td>
            </tr>
          ))}
          {marketplaces.length === 0 && (
            <tr className="border-t border-tertiary">
              <td
                colSpan={7}
                className="p-3 text-sm text-center text-tertiary-alt"
              >
                {t(I18nKey.SETTINGS$MARKETPLACE_ADD_FIRST)}
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}