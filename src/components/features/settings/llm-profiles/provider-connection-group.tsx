import { useId, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { KeyStatusIcon } from "#/components/features/settings/key-status-icon";
import { BrandButton } from "#/components/features/settings/brand-button";
import { EllipsisButton } from "#/components/features/conversation-panel/ellipsis-button";
import { ProviderConnectionActionsMenu } from "./provider-connection-actions-menu";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { cn } from "#/utils/utils";
import {
  settingsListContainerClassName,
  settingsListDividerClassName,
  settingsListIconActionButtonClassName,
} from "#/utils/settings-list-classes";
import { I18nKey } from "#/i18n/declaration";

interface ProviderConnectionGroupProps {
  /** Group heading: the connection's display name, or "Unlinked Profiles". */
  label: string;
  /** Muted text beside the heading (provider id or a short explanation). */
  description?: string | null;
  profileCount: number;
  /**
   * The connection this group represents. Null for the "Unlinked Profiles" group and
   * for profiles whose connection is not in the loaded list; those groups
   * render without key status or a connection menu.
   */
  connection: ProviderConnection | null;
  /** When false, connection actions are hidden (cloud view-only members). */
  canManage: boolean;
  /** Visually separates the "Unlinked Profiles" group from real connections. */
  isUnlinked?: boolean;
  onAddModels?: (connection: ProviderConnection) => void;
  onEdit?: (connection: ProviderConnection) => void;
  onDelete?: (connection: ProviderConnection) => void;
  children: React.ReactNode;
}

/**
 * One provider connection rendered as a collapsible card: a header with the
 * connection's name, provider, linked-profile count, key status and "..."
 * actions, and underneath it the LLM profiles that use the connection.
 */
export function ProviderConnectionGroup({
  label,
  description,
  profileCount,
  connection,
  canManage,
  isUnlinked = false,
  onAddModels,
  onEdit,
  onDelete,
  children,
}: ProviderConnectionGroupProps) {
  const { t } = useTranslation("openhands");
  // Every group starts collapsed so the page opens on a compact list.
  const [isOpen, setIsOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const bodyId = useId();
  const showActions = canManage && connection !== null;

  return (
    <section
      data-testid="provider-connection-group"
      aria-label={label}
      className={cn(
        settingsListContainerClassName,
        "group/connection",
        isUnlinked && "border-dashed bg-transparent",
      )}
    >
      <div
        data-testid={isUnlinked ? undefined : "provider-connection-row"}
        className="flex min-h-12 items-center gap-2 px-2"
      >
        <button
          type="button"
          data-testid="provider-connection-group-toggle"
          aria-expanded={isOpen}
          aria-controls={bodyId}
          onClick={() => setIsOpen((open) => !open)}
          className="flex min-w-0 flex-1 cursor-pointer items-center gap-3 self-stretch rounded-md px-1 text-left"
        >
          <ChevronRight
            aria-hidden
            className={cn(
              "h-4 w-4 shrink-0 text-muted transition-transform duration-75 motion-reduce:transition-none",
              isOpen && "rotate-90",
            )}
          />
          <span
            data-testid="profile-group-header"
            className="min-w-0 max-w-full truncate text-sm font-medium text-contrast"
            title={label}
          >
            {label}
          </span>
          {description ? (
            <span className="min-w-0 max-w-full truncate text-sm text-muted">
              {description}
            </span>
          ) : null}
          <span className="shrink-0 text-sm text-muted">
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_MODEL_COUNT, {
              count: profileCount,
            })}
          </span>
          {connection ? <KeyStatusIcon isSet={connection.api_key_set} /> : null}
        </button>
        {showActions && onAddModels ? (
          // Always shown while the card is expanded; otherwise revealed while
          // it is hovered or holds focus. Touch screens, which have no hover,
          // always show it.
          <BrandButton
            testId="provider-connection-group-add-models"
            type="button"
            variant="secondary"
            onClick={() => onAddModels(connection)}
            className={cn(
              "shrink-0 opacity-0 group-hover/connection:opacity-100 group-focus-within/connection:opacity-100 pointer-coarse:opacity-100",
              isOpen && "opacity-100",
            )}
          >
            {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_ADD_MODELS)}
          </BrandButton>
        ) : null}
        {showActions && (
          <div className="relative shrink-0">
            <EllipsisButton
              ref={triggerRef}
              onClick={() => setMenuOpen((open) => !open)}
              ariaLabel={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_MENU)}
              testId="provider-connection-menu-trigger"
              className={settingsListIconActionButtonClassName}
            />
            {menuOpen && (
              <ProviderConnectionActionsMenu
                anchorRef={triggerRef}
                onAddModels={() => onAddModels?.(connection)}
                onEdit={() => onEdit?.(connection)}
                onDelete={() => onDelete?.(connection)}
                onClose={() => setMenuOpen(false)}
              />
            )}
          </div>
        )}
      </div>

      {isOpen && (
        <div
          id={bodyId}
          data-testid="provider-connection-group-body"
          className={cn("border-t border-border", settingsListDividerClassName)}
        >
          {profileCount > 0 ? (
            children
          ) : (
            <div
              data-testid="provider-connection-group-empty"
              className="flex flex-wrap items-center gap-3 py-3 pl-10 pr-3"
            >
              <span className="text-sm text-muted">
                {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_NO_PROFILES)}
              </span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
