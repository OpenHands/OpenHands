import React from "react";
import { useTranslation } from "react-i18next";
import { SkillWithState } from "#/types/settings";
import { Toggle } from "#/components/shared/toggle/toggle";
import { InfoTooltip } from "#/components/features/settings/info-tooltip";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface SkillsTableProps {
  skills: SkillWithState[];
  onToggle: (skillId: string) => void;
  typeOptions: { key: string; label: string }[];
  repositoryOptions: { key: string; label: string }[];
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onTypeChange: (type: string | null) => void;
  onRepositoryChange: (repo: string | null) => void;
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

export function SkillsTable({
  skills,
  onToggle,
  typeOptions,
  repositoryOptions,
  searchQuery,
  onSearchChange,
  onTypeChange,
  onRepositoryChange,
}: SkillsTableProps) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-4">
      {/* Filters */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
        <input
          data-testid="search-skills-input"
          type="text"
          placeholder={t(I18nKey.SETTINGS$SEARCH_PLACEHOLDER)}
          aria-label={t(I18nKey.SETTINGS$SEARCH_PLACEHOLDER)}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="h-10 w-full flex-1 rounded-sm border border-[#717888] bg-tertiary p-2 placeholder:italic placeholder:text-tertiary-alt focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        />
        <div className="w-full sm:w-52">
          <SettingsDropdownInput
            testId="type-filter-dropdown"
            name="type-filter"
            label={t(I18nKey.SETTINGS$TYPE)}
            items={typeOptions}
            defaultSelectedKey="all"
            onSelectionChange={(key) => onTypeChange(key?.toString() ?? null)}
            placeholder={t(I18nKey.SETTINGS$ALL_TYPES)}
          />
        </div>
        <div className="w-full sm:w-52">
          <SettingsDropdownInput
            testId="repository-filter-dropdown"
            name="repository-filter"
            label={t(I18nKey.SETTINGS$REPOSITORY)}
            items={repositoryOptions}
            defaultSelectedKey="all"
            onSelectionChange={(key) =>
              onRepositoryChange(key?.toString() ?? null)
            }
            placeholder={t(I18nKey.SETTINGS$ALL_REPOSITORIES)}
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-xl border border-tertiary bg-base-secondary/20 table-box-shadow">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] table-fixed border-collapse text-sm">
            <colgroup>
              <col className="w-[28%]" />
              <col className="w-[30%]" />
              <col className="w-[16%]" />
              <col className="w-[16%]" />
              <col className="w-[10%]" />
            </colgroup>
            <thead>
              <tr className="border-b border-tertiary bg-base-secondary/50">
                <th className={HEADER_CLASS}>{t(I18nKey.SETTINGS$NAME)}</th>
                <th className={HEADER_CLASS}>
                  {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
                </th>
                <th className={HEADER_CLASS}>{t(I18nKey.SETTINGS$TYPE)}</th>
                <th className={HEADER_CLASS}>
                  {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
                </th>
                <th className={HEADER_CLASS}>
                  <span className="inline-flex items-center gap-1.5">
                    {t(I18nKey.SETTINGS$ENABLED)}
                    <InfoTooltip
                      content={t(I18nKey.SETTINGS$ENABLED_TOOLTIP)}
                    />
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {skills.map((skill) => (
                <tr
                  key={skill.id}
                  className="border-t border-tertiary/60 transition-colors hover:bg-base-secondary/40"
                >
                  <td
                    className={cn(CELL_CLASS, "font-medium text-content-2")}
                    title={skill.name}
                  >
                    <span className="block truncate">{skill.name}</span>
                  </td>
                  <td className={cn(CELL_CLASS, "text-tertiary-alt")}>
                    <span className="block truncate" title={skill.repository}>
                      {skill.repository}
                    </span>
                  </td>
                  <td className={CELL_CLASS}>
                    <span className="inline-flex items-center rounded-full bg-tertiary/60 px-2.5 py-0.5 text-xs font-medium text-tertiary-light">
                      {skill.type}
                    </span>
                  </td>
                  <td className={CELL_CLASS}>
                    <ScopeBadge scope={skill.scope} />
                  </td>
                  <td className={CELL_CLASS}>
                    {skill.type === "plugin" ? (
                      // Plugins are enabled via their marketplace's Auto-Load
                      // toggle, so this reflects that state read-only.
                      <Toggle
                        checked={skill.isAutoLoad}
                        disabled
                        title={t(I18nKey.SETTINGS$PLUGIN_AUTOLOAD_READONLY)}
                        aria-label={`Auto-load state for ${skill.name}`}
                      />
                    ) : (
                      <Toggle
                        checked={skill.isEnabled}
                        onClick={() => onToggle(skill.id)}
                        aria-label={`Toggle enabled for ${skill.name}`}
                      />
                    )}
                  </td>
                </tr>
              ))}
              {skills.length === 0 && (
                <tr className="border-t border-tertiary/60">
                  <td
                    colSpan={5}
                    className="px-4 py-10 text-center text-sm text-tertiary-alt"
                  >
                    {t(I18nKey.SETTINGS$NO_SKILLS_MATCH_FILTERS)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
