import React from "react";
import { useTranslation } from "react-i18next";
import type { RoutingResolveResult } from "#/api/routing-service/routing-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";

export interface DryRunConsoleProps {
  onResolve: (taskText: string) => void;
  result: RoutingResolveResult | null;
  pending?: boolean;
}

export function DryRunConsole({
  onResolve,
  result,
  pending,
}: DryRunConsoleProps) {
  const { t } = useTranslation("openhands");
  const [task, setTask] = React.useState("");
  return (
    <section data-testid="routing-dry-run" className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$DRY_RUN)}
      </h3>
      <SettingsInput
        testId="routing-dry-run-input"
        name="routing-dry-run-input"
        label={t(I18nKey.ROUTING$TASK_TEXT)}
        type="text"
        value={task}
        onChange={setTask}
      />
      <BrandButton
        type="button"
        variant="primary"
        testId="routing-dry-run-resolve"
        isDisabled={pending || !task.trim()}
        onClick={() => onResolve(task)}
      >
        {t(I18nKey.ROUTING$RESOLVE)}
      </BrandButton>
      {result ? (
        <p data-testid="routing-dry-run-reason" className="text-sm text-white">
          {result.trace.reason}
        </p>
      ) : null}
    </section>
  );
}
