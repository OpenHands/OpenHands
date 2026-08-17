import { I18nKey } from "#/i18n/declaration";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import {
  AGENT_LEVEL_FILTER_OPTIONS,
  type AgentLevelFilter,
} from "./agent-level-filter";

const FILTER_LABEL_KEY: Record<AgentLevelFilter, I18nKey> = {
  all: I18nKey.SETTINGS$AGENTS_LEVEL_ALL,
  builtin: I18nKey.SETTINGS$AGENTS_LEVEL_BUILTIN,
  project: I18nKey.SETTINGS$AGENTS_LEVEL_PROJECT,
  user: I18nKey.SETTINGS$AGENTS_LEVEL_USER,
};

interface AgentsLevelFilterDropdownProps {
  value: AgentLevelFilter;
  onChange: (filter: AgentLevelFilter) => void;
}

export function AgentsLevelFilterDropdown({
  value,
  onChange,
}: AgentsLevelFilterDropdownProps) {
  return (
    <EnumFilterDropdown
      testId="agents-level-filter"
      value={value}
      onChange={onChange}
      options={AGENT_LEVEL_FILTER_OPTIONS}
      labelKeyByValue={FILTER_LABEL_KEY}
    />
  );
}
