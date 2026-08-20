import { useState } from "react";
import { Search, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { usePluginsMarketplace } from "#/hooks/query/use-plugins-marketplace";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
} from "#/utils/extension-module-card-classes";
import { PluginPickerCard } from "./plugin-picker-card";
import {
  isPluginSelected,
  matchesPluginPickerSearch,
  togglePluginSelection,
} from "./plugin-spec-identity";

interface PluginPickerProps {
  /** Currently-attached plugin references (controlled). */
  selected: PluginSpec[];
  /** Called with the next selection whenever the user toggles a plugin. */
  onChange: (next: PluginSpec[]) => void;
  /** Render the toggles read-only (e.g. while a parent submit is in flight). */
  disabled?: boolean;
}

/**
 * Reusable, controlled multi-select over the dynamic plugins catalog
 * (`usePluginsMarketplace`). Selection lives in the parent as `PluginSpec[]`;
 * this component only reads the catalog, filters it, and reports toggles. It is
 * surface-agnostic so the new-conversation flow and the automations UI can both
 * embed it. No install/enable/disable — that is the plugins management page.
 */
export function PluginPicker({
  selected,
  onChange,
  disabled = false,
}: PluginPickerProps) {
  const { t } = useTranslation("openhands");
  const { data: catalog, isLoading, isError } = usePluginsMarketplace();
  const [search, setSearch] = useState("");

  const visible = (catalog ?? []).filter((plugin) =>
    matchesPluginPickerSearch(plugin, search),
  );

  const renderBody = () => {
    if (isLoading) {
      return (
        <p
          data-testid="plugin-picker-loading"
          className="py-8 text-center text-sm text-tertiary-light"
        >
          {t(I18nKey.PLUGINS$PICKER_LOADING)}
        </p>
      );
    }
    if (isError) {
      return (
        <p
          data-testid="plugin-picker-error"
          className="py-8 text-center text-sm text-tertiary-light"
        >
          {t(I18nKey.PLUGINS$PICKER_ERROR)}
        </p>
      );
    }
    if (visible.length === 0) {
      return (
        <p
          data-testid="plugin-picker-empty"
          className="py-8 text-center text-sm text-tertiary-light"
        >
          {t(I18nKey.PLUGINS$PICKER_EMPTY)}
        </p>
      );
    }
    return (
      <div className={extensionModuleCardGridContainerClassName}>
        <div className={extensionModuleCardGridClassName}>
          {visible.map((plugin) => (
            <PluginPickerCard
              key={`${plugin.source} ${plugin.ref ?? ""} ${plugin.repo_path ?? ""} ${plugin.name}`}
              plugin={plugin}
              isSelected={isPluginSelected(selected, plugin)}
              isDisabled={disabled}
              onToggle={() => onChange(togglePluginSelection(selected, plugin))}
            />
          ))}
        </div>
      </div>
    );
  };

  return (
    <div data-testid="plugin-picker" className="flex w-full flex-col gap-3">
      <div
        className={cn(
          "relative flex min-w-0 items-center",
          "rounded-lg border border-[var(--oh-border)] bg-base-secondary",
          "transition-colors focus-within:border-white/40 focus-within:ring-1 focus-within:ring-white/20",
        )}
      >
        <Search
          className="ml-3 h-4 w-4 shrink-0 text-tertiary-alt"
          aria-hidden
        />
        <input
          data-testid="plugin-picker-search-input"
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder={t(I18nKey.PLUGINS$PICKER_SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.PLUGINS$PICKER_SEARCH_PLACEHOLDER)}
          className={cn(
            "min-w-0 flex-1 border-0 bg-transparent outline-none",
            "px-3 py-2 text-sm placeholder:text-tertiary-alt",
            "[&::-webkit-search-cancel-button]:hidden",
          )}
        />
        {search ? (
          <button
            type="button"
            onClick={() => setSearch("")}
            aria-label={t(I18nKey.MCP$SEARCH_CLEAR)}
            className="mr-2 cursor-pointer rounded p-1 text-tertiary-alt hover:text-white"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        ) : null}
      </div>

      {renderBody()}
    </div>
  );
}
