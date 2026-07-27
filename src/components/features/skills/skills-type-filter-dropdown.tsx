import { I18nKey } from "#/i18n/declaration";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import {
  SKILL_TYPE_FILTER_OPTIONS,
  type SkillTypeFilter,
} from "./skill-type-filter";

const FILTER_LABEL_KEY: Record<SkillTypeFilter, I18nKey> = {
  all: I18nKey.SETTINGS$SKILLS_TYPE_ALL,
  agentskills: I18nKey.SETTINGS$SKILLS_TYPE_AGENTSKILLS,
  knowledge: I18nKey.SETTINGS$SKILLS_TYPE_KNOWLEDGE,
  repo: I18nKey.SETTINGS$SKILLS_TYPE_REPO,
};

interface SkillsTypeFilterDropdownProps {
  value: SkillTypeFilter;
  onChange: (filter: SkillTypeFilter) => void;
}

export function SkillsTypeFilterDropdown({
  value,
  onChange,
}: SkillsTypeFilterDropdownProps) {
  return (
    <EnumFilterDropdown
      testId="skills-type-filter"
      value={value}
      onChange={onChange}
      options={SKILL_TYPE_FILTER_OPTIONS}
      labelKeyByValue={FILTER_LABEL_KEY}
    />
  );
}
