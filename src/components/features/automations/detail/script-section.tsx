import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import TerminalIcon from "#/icons/terminal.svg?react";
import type { Automation } from "#/types/automation";
import { AutomationBundleFiles } from "./automation-bundle-files";
import { SectionCard } from "./section-card";

interface ScriptSectionProps {
  automation: Automation;
}

/**
 * What a script automation runs: its entrypoint and the files of its
 * uploaded bundle. Takes the place of the prompt section for automations
 * that have no prompt.
 */
export function ScriptSection({ automation }: ScriptSectionProps) {
  const { t } = useTranslation("openhands");

  return (
    <SectionCard
      icon={<TerminalIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT)}
    >
      <div className="flex flex-col gap-4">
        {automation.entrypoint ? (
          <code
            data-testid="automation-script-entrypoint"
            className="self-start rounded-md bg-surface-raised px-2 py-1 font-mono text-xs text-content"
          >
            {automation.entrypoint}
          </code>
        ) : null}

        <AutomationBundleFiles automation={automation} />
      </div>
    </SectionCard>
  );
}
