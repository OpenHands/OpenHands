import React from "react";
import { useTranslation } from "react-i18next";
import type { ChannelRecord } from "#/api/channel-service/channel-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import { useUpdateChannelConfig } from "#/hooks/query/use-channels";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";

interface ChannelConfigProps {
  channel: ChannelRecord;
  isOpen: boolean;
  onClose: () => void;
}

export function ChannelConfig({
  channel,
  isOpen,
  onClose,
}: ChannelConfigProps) {
  const { t } = useTranslation("openhands");
  const updateConfig = useUpdateChannelConfig();
  const [humanOnly, setHumanOnly] = React.useState(
    Boolean(channel.config.human_only),
  );
  const [costCap, setCostCap] = React.useState(
    channel.config.cost_cap != null ? String(channel.config.cost_cap) : "",
  );
  const [rulesText, setRulesText] = React.useState(
    JSON.stringify(channel.config.routing_rules ?? [], null, 2),
  );

  React.useEffect(() => {
    setHumanOnly(Boolean(channel.config.human_only));
    setCostCap(
      channel.config.cost_cap != null ? String(channel.config.cost_cap) : "",
    );
    setRulesText(JSON.stringify(channel.config.routing_rules ?? [], null, 2));
  }, [channel]);

  const onSave = async () => {
    let routing_rules = channel.config.routing_rules ?? [];
    try {
      routing_rules = JSON.parse(rulesText) as typeof routing_rules;
    } catch {
      routing_rules = channel.config.routing_rules ?? [];
    }
    await updateConfig.mutateAsync({
      channelId: channel.id,
      payload: {
        human_only: humanOnly,
        cost_cap: costCap === "" ? null : Number(costCap),
        routing_rules,
      },
    });
    onClose();
  };

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.CHANNELS$CONFIG)}
      onClose={onClose}
      footer={
        <div className="flex justify-end gap-2">
          <BrandButton type="button" variant="tertiary" onClick={onClose}>
            {t(I18nKey.CHANNELS$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            testId={`channel-config-save-${channel.id}`}
            onClick={() => {
              void onSave();
            }}
          >
            {t(I18nKey.CHANNELS$SAVE)}
          </BrandButton>
        </div>
      }
    >
      <label className="mb-3 block text-sm text-white">
        {t(I18nKey.CHANNELS$ROUTING_RULES)}
        <textarea
          data-testid={`channel-config-rules-${channel.id}`}
          className="mt-1 w-full rounded bg-base-secondary p-2 font-mono text-xs"
          rows={6}
          value={rulesText}
          onChange={(event) => setRulesText(event.target.value)}
        />
      </label>
      <label className="mb-3 block text-sm text-white">
        {t(I18nKey.CHANNELS$COST_CAP)}
        <input
          data-testid={`channel-config-cost-cap-${channel.id}`}
          className="mt-1 w-full rounded bg-base-secondary p-2"
          value={costCap}
          onChange={(event) => setCostCap(event.target.value)}
        />
      </label>
      <div data-testid={`channel-config-human-only-${channel.id}`}>
        <ToggleSwitch
          enabled={humanOnly}
          label={t(I18nKey.CHANNELS$HUMAN_ONLY)}
          onToggle={() => setHumanOnly((value) => !value)}
        />
      </div>
    </ApiKeyModalBase>
  );
}
