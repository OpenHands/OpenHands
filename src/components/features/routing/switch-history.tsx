import { useTranslation } from "react-i18next";
import type { RoutingAuditItem } from "#/api/routing-service/routing-types";
import { CostText } from "#/components/shared/cost-text";
import { I18nKey } from "#/i18n/declaration";

export interface SwitchHistoryProps {
  items: RoutingAuditItem[];
}

function targetLabel(
  target: { provider_key?: string | null; model?: string | null } | undefined,
): string {
  if (!target?.provider_key && !target?.model) {
    return "";
  }
  return `${target.provider_key ?? ""}/${target.model ?? ""}`;
}

export function SwitchHistory({ items }: SwitchHistoryProps) {
  const { t } = useTranslation("openhands");
  const switches = items.filter((item) => item.kind === "switch");
  const resolves = items.filter((item) => item.kind === "resolve");

  if (items.length === 0) {
    return (
      <p
        data-testid="routing-audit-empty"
        className="text-sm text-tertiary-light"
      >
        {t(I18nKey.ROUTING$AUDIT_EMPTY)}
      </p>
    );
  }

  return (
    <section
      data-testid="routing-switch-history"
      className="flex flex-col gap-2"
    >
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$SWITCH_HISTORY)}
      </h3>
      <ul className="flex flex-col gap-2">
        {switches.map((item) => (
          <li
            key={item.id}
            data-testid={`routing-switch-${item.id}`}
            className="rounded-xl bg-base-secondary p-3 text-sm text-white"
          >
            <p data-testid={`routing-switch-from-${item.id}`}>
              {t(I18nKey.ROUTING$SWITCH_FROM)}: {targetLabel(item.payload.from)}
            </p>
            <p data-testid={`routing-switch-to-${item.id}`}>
              {t(I18nKey.ROUTING$SWITCH_TO)}: {targetLabel(item.payload.to)}
            </p>
            <p data-testid={`routing-switch-reason-${item.id}`}>
              {t(I18nKey.ROUTING$SWITCH_REASON)}:{" "}
              {item.payload.reason === "struggle"
                ? t(I18nKey.ROUTING$STRUGGLE)
                : (item.payload.reason ?? "")}
            </p>
          </li>
        ))}
        {resolves.map((item) => {
          const cost = item.payload.trace?.ranked?.[0]?.cost_per_1k;
          return (
            <li
              key={item.id}
              data-testid={`routing-audit-resolve-${item.id}`}
              className="rounded-xl bg-base-secondary p-3 text-sm text-white"
            >
              <p>
                {item.payload.decision?.provider_key}/
                {item.payload.decision?.model}
              </p>
              {typeof cost === "number" ? (
                <p data-testid={`routing-audit-cost-${item.id}`}>
                  {t(I18nKey.ROUTING$COST_PER_TASK)}:{" "}
                  <CostText amount={cost} at={item.created_at} />
                </p>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
