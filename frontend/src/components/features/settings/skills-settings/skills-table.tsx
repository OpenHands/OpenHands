import React from "react";
import { useTranslation } from "react-i18next";
import { SkillWithState } from "#/types/settings";
import { Toggle } from "#/components/shared/toggle/toggle";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

interface SkillsTableProps {
  skills: SkillWithState[];
  onToggle: (skillId: string) => void;
  typeOptions: { key: string; label: string }[];
  repositoryOptions: { key: string; label: string }[];
  searchQuery: string;
  selectedType: string | null;
  selectedRepository: string | null;
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

export function SkillsTable({
  skills,
  onToggle,
  typeOptions,
  repositoryOptions,
  searchQuery,
  selectedType,
  selectedRepository,
  onSearchChange,
  onTypeChange,
  onRepositoryChange,
}: SkillsTableProps) {
  const { t } = useTranslation();

  return (
    <>
      {/* Filters */}
      <div className="flex items-stretch gap-4 justify-center">
        <div className="flex-1 flex flex-col gap-2.5">
          <div className="h-5" />
          <input
            data-testid="search-skills-input"
            type="text"
            placeholder={t(I18nKey.SETTINGS$SEARCH_PLACEHOLDER)}
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt"
          />
        </div>
        <div className="flex-1">
          <SettingsDropdownInput
            testId="type-filter-dropdown"
            name="type-filter"
            label="TYPE"
            items={typeOptions}
            defaultSelectedKey="all"
            onSelectionChange={(key) => onTypeChange(key?.toString() ?? null)}
            placeholder={t(I18nKey.SETTINGS$ALL_TYPES)}
          />
        </div>
        <div className="flex-1">
          <SettingsDropdownInput
            testId="repository-filter-dropdown"
            name="repository-filter"
            label="REPOSITORY"
            items={repositoryOptions}
            defaultSelectedKey="all"
            onSelectionChange={(key) => onRepositoryChange(key?.toString() ?? null)}
            placeholder={t(I18nKey.SETTINGS$ALL_REPOSITORIES)}
          />
        </div>
      </div>

      {/* Table */}
      <div className="border border-tertiary rounded-md overflow-hidden">
        <table className="w-full">
          <thead className="bg-base-secondary">
            <tr className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] gap-4 items-start">
              <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                {t(I18nKey.SETTINGS$NAME)}
              </th>
              <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
              </th>
              <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                {t(I18nKey.SETTINGS$TYPE)}
              </th>
              <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
              </th>
              <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                {t(I18nKey.SETTINGS$ENABLED)}
              </th>
            </tr>
          </thead>
          <tbody>
            {skills.map((skill) => (
              <tr
                key={skill.id}
                className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] gap-4 items-center border-t border-tertiary"
              >
                <td className="p-3 text-sm text-content-2 truncate min-w-0">
                  {skill.name}
                </td>
                <td className="p-3 text-sm text-tertiary-alt truncate">
                  {skill.repository}
                </td>
                <td className="p-3">
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-base-secondary text-tertiary-alt">
                    {skill.type}
                  </span>
                </td>
                <td className="p-3">
                  <ScopeBadge scope={skill.scope} />
                </td>
                <td className="p-3">
                  <Toggle
                    checked={skill.isEnabled}
                    onClick={() => onToggle(skill.id)}
                    aria-label={`Toggle enabled for ${skill.name}`}
                  />
                </td>
              </tr>
            ))}
            {skills.length === 0 && (
              <tr className="border-t border-tertiary">
                <td
                  colSpan={5}
                  className="p-3 text-sm text-center text-[rgb(140,140,140)]"
                >
                  {t(I18nKey.SETTINGS$NO_SKILLS_MATCH_FILTERS)}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}