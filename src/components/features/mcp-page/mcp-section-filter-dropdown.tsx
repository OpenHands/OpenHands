import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { EnumFilterDropdown } from "#/components/shared/filters/enum-filter-dropdown";
import {
  MCP_SECTION_FILTER_OPTIONS,
  type McpSectionFilter,
} from "./mcp-section-filter";

const FILTER_LABEL_KEY: Record<McpSectionFilter, I18nKey> = {
  all: I18nKey.MCP$SECTION_FILTER_ALL,
  installed: I18nKey.MCP$INSTALLED_TITLE,
  library: I18nKey.MCP$LIBRARY_TITLE,
};

interface McpSectionFilterDropdownProps {
  value: McpSectionFilter;
  onChange: (filter: McpSectionFilter) => void;
}

export function McpSectionFilterDropdown({
  value,
  onChange,
}: McpSectionFilterDropdownProps) {
  const { t } = useTranslation("openhands");

  return (
    <EnumFilterDropdown
      testId="mcp-section-filter"
      value={value}
      onChange={onChange}
      options={MCP_SECTION_FILTER_OPTIONS}
      labelKeyByValue={FILTER_LABEL_KEY}
      ariaLabel={t(I18nKey.MCP$SECTION_FILTER_LABEL)}
    />
  );
}
