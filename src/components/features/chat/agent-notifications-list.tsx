import { useEffect, useId, useMemo, useState } from "react";
import { ChevronDown, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import ClockIcon from "#/icons/clock.svg?react";
import SkillsIcon from "#/icons/skills.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { formControlButtonCompactClassName } from "#/utils/form-control-classes";
import { AgentNotificationKindPill } from "./agent-notification-kind-pill";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type {
  AgentNotification,
  AgentNotificationKind,
} from "./agent-notifications.constants";

const CHECKBOX_CLASSNAME =
  "size-4 shrink-0 rounded border-[var(--oh-border)] bg-base-secondary";

const ROW_ACTION_BUTTON_CLASSNAME = cn(
  "inline-flex size-6 shrink-0 items-center justify-center rounded-sm",
  "text-muted transition-colors hover:bg-white/10 hover:text-white",
);

const ROW_REMOVE_BUTTON_CLASSNAME = cn(
  ROW_ACTION_BUTTON_CLASSNAME,
  "opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100",
);

const ROW_KIND_ICON_CLASSNAME = "size-4 shrink-0 text-[var(--oh-muted)]";

function AgentNotificationKindIcon({ kind }: { kind: AgentNotificationKind }) {
  if (kind === "skill") {
    return <SkillsIcon className={ROW_KIND_ICON_CLASSNAME} aria-hidden />;
  }

  return <ClockIcon className={ROW_KIND_ICON_CLASSNAME} aria-hidden />;
}

interface AgentNotificationsListProps {
  agentNotifications: AgentNotification[];
  onSubmit: (selectedIds: string[]) => void;
  onRemove?: (id: string) => void;
  onDismiss?: () => void;
  disabled?: boolean;
  isSubmitting?: boolean;
  submitTestId?: string;
  dismissTestId?: string;
  listItemTestIdPrefix?: string;
}

interface AgentNotificationListItemProps {
  agentNotification: AgentNotification;
  isChecked: boolean;
  isExpanded: boolean;
  controlsDisabled: boolean;
  listItemTestIdPrefix: string;
  onToggleChecked: () => void;
  onToggleExpanded: () => void;
  onRemove?: (id: string) => void;
}

function AgentNotificationListItem({
  agentNotification,
  isChecked,
  isExpanded,
  controlsDisabled,
  listItemTestIdPrefix,
  onToggleChecked,
  onToggleExpanded,
  onRemove,
}: AgentNotificationListItemProps) {
  const { t } = useTranslation("openhands");
  const detailsId = useId();
  const checkboxId = `${listItemTestIdPrefix}-${agentNotification.id}`;

  return (
    <li className="group">
      <div
        data-testid={`${listItemTestIdPrefix}-${agentNotification.id}`}
        data-state={isExpanded ? "expanded" : "collapsed"}
        className={cn(
          "flex items-center gap-3 px-3 py-2 transition-colors",
          isChecked && "bg-[var(--oh-surface)]",
          !controlsDisabled && "hover:bg-[var(--oh-surface)]",
        )}
      >
        <input
          id={checkboxId}
          type="checkbox"
          data-testid={`${listItemTestIdPrefix}-checkbox-${agentNotification.id}`}
          checked={isChecked}
          disabled={controlsDisabled}
          onChange={onToggleChecked}
          aria-label={agentNotification.name}
          className={CHECKBOX_CLASSNAME}
        />
        <label
          htmlFor={checkboxId}
          className={cn(
            "flex min-w-0 flex-1 items-center gap-2",
            controlsDisabled ? "cursor-not-allowed" : "cursor-pointer",
          )}
        >
          <AgentNotificationKindIcon kind={agentNotification.kind} />
          <span className="min-w-0 truncate text-sm font-medium text-content">
            {agentNotification.name}
          </span>
          <AgentNotificationKindPill
            kind={agentNotification.kind}
            testId={`${listItemTestIdPrefix}-kind-pill-${agentNotification.id}`}
          />
        </label>
        {onRemove ? (
          <button
            type="button"
            data-testid={`${listItemTestIdPrefix}-remove-${agentNotification.id}`}
            aria-label={t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_REMOVE, {
              name: agentNotification.name,
            })}
            disabled={controlsDisabled}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRemove(agentNotification.id);
            }}
            className={cn(
              ROW_REMOVE_BUTTON_CLASSNAME,
              controlsDisabled && "cursor-not-allowed opacity-50",
            )}
          >
            <Trash2 className="size-4" strokeWidth={2} aria-hidden />
          </button>
        ) : null}
        <button
          type="button"
          data-testid={`${listItemTestIdPrefix}-expand-${agentNotification.id}`}
          aria-expanded={isExpanded}
          aria-controls={detailsId}
          aria-label={
            isExpanded
              ? t(I18nKey.BUTTON$COLLAPSE_DETAILS)
              : t(I18nKey.BUTTON$EXPAND_DETAILS)
          }
          disabled={controlsDisabled}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onToggleExpanded();
          }}
          className={cn(
            ROW_ACTION_BUTTON_CLASSNAME,
            controlsDisabled && "cursor-not-allowed opacity-50",
          )}
        >
          <ChevronDown
            className={cn(
              "size-4 transition-transform duration-200 motion-reduce:transition-none",
              isExpanded && "rotate-180",
            )}
            aria-hidden
          />
        </button>
      </div>
      {isExpanded ? (
        <div
          id={detailsId}
          data-testid={`${listItemTestIdPrefix}-details-${agentNotification.id}`}
          className="border-t border-[var(--oh-border)] bg-[var(--oh-surface)]/50 px-3 py-2 pl-11"
        >
          <p className="max-h-32 overflow-y-auto break-words text-xs text-muted custom-scrollbar">
            {agentNotification.prompt}
          </p>
        </div>
      ) : null}
    </li>
  );
}

export function AgentNotificationsList({
  agentNotifications,
  onSubmit,
  onRemove,
  onDismiss,
  disabled = false,
  isSubmitting = false,
  submitTestId = "agent-notifications-submit",
  dismissTestId = "agent-notifications-dismiss-action",
  listItemTestIdPrefix = "agent-notification",
}: AgentNotificationsListProps) {
  const { t } = useTranslation("openhands");
  const agentNotificationIds = useMemo(
    () => agentNotifications.map((agentNotification) => agentNotification.id),
    [agentNotifications],
  );
  const [selectedIds, setSelectedIds] =
    useState<string[]>(agentNotificationIds);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    setSelectedIds(agentNotificationIds);
  }, [agentNotificationIds]);

  useEffect(() => {
    setExpandedIds((current) => {
      const validIds = new Set(agentNotificationIds);
      const next = new Set([...current].filter((id) => validIds.has(id)));
      return next.size === current.size ? current : next;
    });
  }, [agentNotificationIds]);

  const selectedIdSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const hasSelection = selectedIds.length > 0;
  const controlsDisabled = disabled || isSubmitting;

  const toggleAgentNotification = (id: string) => {
    setSelectedIds((current) =>
      current.includes(id)
        ? current.filter((entry) => entry !== id)
        : [...current, id],
    );
  };

  const toggleExpanded = (id: string) => {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleSubmit = () => {
    if (!hasSelection || controlsDisabled) {
      return;
    }
    onSubmit(selectedIds);
  };

  if (agentNotifications.length === 0) {
    return null;
  }

  return (
    <>
      <ul
        className={cn(
          "flex flex-col divide-y divide-[var(--oh-border)] rounded-md border",
          "border-[var(--oh-border)] overflow-hidden",
        )}
      >
        {agentNotifications.map((agentNotification) => (
          <AgentNotificationListItem
            key={agentNotification.id}
            agentNotification={agentNotification}
            isChecked={selectedIdSet.has(agentNotification.id)}
            isExpanded={expandedIds.has(agentNotification.id)}
            controlsDisabled={controlsDisabled}
            listItemTestIdPrefix={listItemTestIdPrefix}
            onToggleChecked={() =>
              toggleAgentNotification(agentNotification.id)
            }
            onToggleExpanded={() => toggleExpanded(agentNotification.id)}
            onRemove={onRemove}
          />
        ))}
      </ul>

      <div className="mt-4 flex justify-end gap-2">
        {onDismiss ? (
          <BrandButton
            testId={dismissTestId}
            type="button"
            variant="secondary"
            isDisabled={controlsDisabled}
            className={formControlButtonCompactClassName}
            onClick={onDismiss}
          >
            {t(I18nKey.CHAT_INTERFACE$MESSAGE_DISMISS)}
          </BrandButton>
        ) : null}
        <BrandButton
          testId={submitTestId}
          type="button"
          variant="primary"
          isDisabled={!hasSelection || controlsDisabled}
          aria-busy={isSubmitting}
          className={formControlButtonCompactClassName}
          onClick={handleSubmit}
        >
          {t(I18nKey.CHAT_INTERFACE$AGENT_NOTIFICATIONS_CREATE_ALL)}
        </BrandButton>
      </div>
    </>
  );
}
