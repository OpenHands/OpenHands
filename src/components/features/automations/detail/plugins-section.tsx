import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import PuzzleIcon from "#/icons/puzzle.svg?react";
import { SectionCard } from "./section-card";
import { PluginChip } from "./plugin-chip";

interface PluginsSectionProps {
  plugins: string[];
}

export function PluginsSection({ plugins }: PluginsSectionProps) {
  const { t } = useTranslation("openhands");

  return (
    <SectionCard
      icon={<PuzzleIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$PLUGINS)}
    >
      <div className="flex flex-wrap gap-2">
        {plugins.map((plugin) => (
          <PluginChip key={plugin} name={plugin} />
        ))}
      </div>
    </SectionCard>
  );
}
