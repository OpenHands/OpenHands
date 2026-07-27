import React from "react";
import { useTranslation } from "react-i18next";
import { useMatch, useNavigate } from "react-router";
import { Plus, Settings } from "lucide-react";
import { Dropdown } from "#/ui/dropdown/dropdown";
import { DropdownOption } from "#/ui/dropdown/types";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";
import {
  useBackendsHealth,
  type BackendHealth,
} from "#/hooks/query/use-backends-health";
import { I18nKey } from "#/i18n/declaration";
import type { Backend } from "#/api/backend-registry/types";
// Import the trigger helpers from the lightweight store, not the overlay
// component, so the eagerly-mounted sidebar/backend-selector graph does not
// pull in the overlay's render code (the overlay is lazy-loaded from
// `routes/root-layout.tsx`).
import {
  ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS,
  triggerEnvironmentSwitch,
} from "#/components/features/backends/environment-switch-store";
import { NavigationLink } from "#/components/shared/navigation-link";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { useConversationStore } from "#/stores/conversation-store";
import { AddBackendModal } from "./add-backend-modal";
import { BackendStatusDot } from "./backend-status-dot";
import { ManageBackendsModal } from "./manage-backends-modal";
import { cn } from "#/utils/utils";
import { formControlTransitionClassName } from "#/utils/form-control-classes";
import {
  dropdownFooterActionClassName,
  dropdownMenuListClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";

const VALUE_SEPARATOR = "::";

function makeOptionValue(backendId: string, orgId: string | null): string {
  return orgId ? `${backendId}${VALUE_SEPARATOR}${orgId}` : backendId;
}

function parseOptionValue(value: string): {
  backendId: string;
  orgId: string | null;
} {
  const [backendId, orgId] = value.split(VALUE_SEPARATOR);
  return { backendId, orgId: orgId ?? null };
}

function buildStatusPrefix(health: BackendHealth | undefined) {
  return <BackendStatusDot isConnected={health?.isConnected ?? null} />;
}

function buildNoBackendPrefix() {
  return <BackendStatusDot isConnected="unavailable" />;
}

function buildOptions(
  registered: Backend[],
  personalWorkspaceLabel: string,
  cloudOrgs: ReturnType<typeof useAllCloudOrganizations>,
  currentUserIds: ReturnType<typeof useCloudCurrentUserId>,
  healthByBackendId: Record<string, BackendHealth>,
): DropdownOption[] {
  const options: DropdownOption[] = [];

  const locals = registered.filter((b) => b.kind === "local");
  const clouds = registered.filter((b) => b.kind === "cloud");

  for (const b of locals) {
    options.push({
      value: makeOptionValue(b.id, null),
      label: b.name,
      prefix: buildStatusPrefix(healthByBackendId[b.id]),
    });
  }

  for (const b of clouds) {
    const entry = cloudOrgs[b.id];
    const prefix = buildStatusPrefix(healthByBackendId[b.id]);
    if (!entry || entry.orgs.length === 0) {
      options.push({
        value: makeOptionValue(b.id, null),
        label: b.name,
        prefix,
      });
    } else {
      // Personal-workspace rule (per the cloud contract): the org whose
      // id matches the calling user's id is the user's personal
      // workspace. We resolve `user_id` once per backend (via /me on any
      // one org) and apply it across all orgs of that backend.
      const userIdForBackend = currentUserIds[b.id]?.userId ?? null;

      for (const org of entry.orgs) {
        const isPersonal = !!userIdForBackend && userIdForBackend === org.id;
        const orgLabel = isPersonal ? personalWorkspaceLabel : org.name;
        options.push({
          value: makeOptionValue(b.id, org.id),
          label: `${b.name} – ${orgLabel}`,
          // All org rows for the same cloud backend share that backend's
          // single connectivity verdict — there is no per-org probe.
          prefix,
        });
      }
    }
  }

  return options;
}

interface BackendSelectorProps {
  /** Render the menu above the trigger (e.g. when pinned to bottom of sidebar). */
  openUpward?: boolean;
  /** Hide the selector input trigger and only render the dropdown menu. */
  hideTrigger?: boolean;
  /** Whether the dropdown menu should start open on mount. */
  defaultOpen?: boolean;
  /** Callback fired after selecting a backend/org option. */
  onSelectOption?: () => void;
  /**
   * Override the internal Add Backend modal handling. When provided,
   * clicking "Add Backend" calls this instead of opening BackendSelector's
   * own modal. Useful when the selector is mounted inside an ephemeral
   * container (e.g. the collapsed-sidebar popover) and the modal must
   * survive the parent unmounting.
   */
  onOpenAddBackend?: () => void;
  /** Same as onOpenAddBackend but for the Manage Backends modal. */
  onOpenManageBackends?: () => void;
  /**
   * Whether the surrounding sidebar rail is in its collapsed variant. Passed
   * down from `SidebarRailBody` so the mobile drawer (which always renders
   * the expanded rail) can override the persisted desktop value.
   */
  sidebarCollapsed?: boolean;
}

export function BackendSelector({
  openUpward = false,
  hideTrigger = false,
  defaultOpen = false,
  onSelectOption,
  onOpenAddBackend,
  onOpenManageBackends,
  sidebarCollapsed = false,
}: BackendSelectorProps = {}) {
  const { t } = useTranslation("openhands");
  const { backends, active, setActive } = useActiveBackendContext();
  const cloudOrgs = useAllCloudOrganizations();
  const currentUserIds = useCloudCurrentUserId();
  // Probe each registered backend every 10s.
  const healthByBackendId = useBackendsHealth(backends);
  const navigate = useNavigate();
  const settingsMatch = useMatch("/settings");
  const settingsSubrouteMatch = useMatch("/settings/*");
  const conversationMatch = useMatch("/conversations/:conversationId");
  const automationDetailMatch = useMatch("/automations/:automationId");
  const [addBackendModalOpen, setAddBackendModalOpen] = React.useState(false);
  const [manageBackendsModalOpen, setManageBackendsModalOpen] =
    React.useState(false);

  const personalWorkspaceLabel = t(I18nKey.BACKEND$PERSONAL_WORKSPACE);

  const options = React.useMemo(
    () =>
      buildOptions(
        backends,
        personalWorkspaceLabel,
        cloudOrgs,
        currentUserIds,
        healthByBackendId,
      ),
    [
      backends,
      personalWorkspaceLabel,
      cloudOrgs,
      currentUserIds,
      healthByBackendId,
    ],
  );

  const noBackendSelected = isNoBackend(active.backend);
  const noBackendLabel = t(I18nKey.BACKEND$NO_BACKEND_AVAILABLE);
  const activeValue = makeOptionValue(active.backend.id, active.orgId);
  const activeOption = noBackendSelected
    ? undefined
    : options.find((o) => o.value === activeValue);
  const isSettingsActive = Boolean(settingsMatch || settingsSubrouteMatch);
  const settingsLabel = t(I18nKey.SIDEBAR$SETTINGS);
  const isRightPanelShown = useConversationStore(
    (state) => state.isRightPanelShown,
  );
  // When the sidebar rail is expanded, `placement="left"` hugs the main
  // canvas and reads awkwardly; prefer above the control. When the rail is
  // collapsed, keep left except on active conversation + open right drawer.
  const settingsTooltipPlacement =
    !sidebarCollapsed || (conversationMatch && isRightPanelShown)
      ? "top"
      : "left";

  const someCloudLoading = Object.values(cloudOrgs).some((c) => c.isLoading);

  // Self-heal a malformed `(cloudBackendId, null)` selection.
  //
  // Once a cloud backend's orgs resolve, the dropdown only renders
  // per-org rows for it — the `(backendId, null)` row disappears, so
  // selecting that shape would drift from what the dropdown can render
  // (UI says "Local", APIs hit cloud). When we detect the drift, snap
  // the selection onto Cloud's current org first, then fall back to the
  // personal workspace (or, lacking a /me result, the first org). The
  // selection is recorded locally only; the cloud request scope follows
  // from the X-Org-Id header sent by `callCloudProxy`, so the cloud UI's
  // org choice is never mutated as a side effect.
  React.useEffect(() => {
    if (noBackendSelected || active.backend.kind !== "cloud" || active.orgId)
      return;
    const { backend } = active;
    const entry = cloudOrgs[backend.id];
    if (!entry || entry.orgs.length === 0) return;

    const currentOrg = entry.currentOrgId
      ? entry.orgs.find((o) => o.id === entry.currentOrgId)
      : undefined;
    const userId = currentUserIds[backend.id]?.userId ?? null;
    const personal = userId
      ? entry.orgs.find((o) => o.id === userId)
      : undefined;
    const target = currentOrg ?? personal ?? entry.orgs[0];
    if (target) {
      setActive(backend.id, target.id);
    }
  }, [active, cloudOrgs, currentUserIds, setActive, noBackendSelected]);

  const openAddBackendModal = React.useCallback(() => {
    if (onOpenAddBackend) {
      onOpenAddBackend();
      onSelectOption?.();
      return;
    }
    setAddBackendModalOpen(true);
  }, [onOpenAddBackend, onSelectOption]);

  const openManageBackendsModal = React.useCallback(() => {
    if (onOpenManageBackends) {
      onOpenManageBackends();
      onSelectOption?.();
      return;
    }
    setManageBackendsModalOpen(true);
  }, [onOpenManageBackends, onSelectOption]);

  const isLockedToCloud = getLockedCloudHost() !== null;

  const preventDropdownMenuClose = React.useCallback(
    (event: React.SyntheticEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
    },
    [],
  );

  const handleAddBackendClick = React.useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      openAddBackendModal();
    },
    [openAddBackendModal, preventDropdownMenuClose],
  );

  const handleAddBackendTouchEnd = React.useCallback(
    (event: React.TouchEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      openAddBackendModal();
    },
    [openAddBackendModal, preventDropdownMenuClose],
  );

  const handleManageBackendsClick = React.useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      openManageBackendsModal();
    },
    [openManageBackendsModal, preventDropdownMenuClose],
  );

  const handleManageBackendsTouchEnd = React.useCallback(
    (event: React.TouchEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      openManageBackendsModal();
    },
    [openManageBackendsModal, preventDropdownMenuClose],
  );

  const addBackendFooter = (
    <div className={dropdownMenuListClassName}>
      {isLockedToCloud ? null : (
        <button
          type="button"
          data-testid="add-backend-menu-item"
          onMouseDown={preventDropdownMenuClose}
          onTouchStart={preventDropdownMenuClose}
          onTouchEnd={handleAddBackendTouchEnd}
          onClick={handleAddBackendClick}
          className={cn(
            dropdownFooterActionClassName,
            "cursor-pointer rounded-md",
          )}
        >
          <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
            <Plus width={16} height={16} />
          </span>
          {t(I18nKey.BACKEND$ADD)}
        </button>
      )}
      <button
        type="button"
        data-testid="manage-backends-menu-item"
        onMouseDown={preventDropdownMenuClose}
        onTouchStart={preventDropdownMenuClose}
        onTouchEnd={handleManageBackendsTouchEnd}
        onClick={handleManageBackendsClick}
        className={cn(
          dropdownFooterActionClassName,
          "cursor-pointer rounded-md",
        )}
      >
        <span className={dropdownMenuRowIconWrapperClassName} aria-hidden>
          <Settings width={16} height={16} />
        </span>
        {t(
          isLockedToCloud
            ? I18nKey.BACKEND$RECONNECT_CLOUD
            : I18nKey.BACKEND$MANAGE,
        )}
      </button>
    </div>
  );

  const handleSelectBackend = React.useCallback(
    async (value: string) => {
      if (value === activeValue) return;

      const { backendId, orgId } = parseOptionValue(value);
      const target = backends.find((b) => b.id === backendId);
      if (!target) return;

      triggerEnvironmentSwitch(
        options.find((option) => option.value === value)?.label ?? target.name,
      );
      await new Promise<void>((resolve) => {
        setTimeout(resolve, ENVIRONMENT_SWITCH_SETACTIVE_DELAY_MS);
      });

      // @spec BM-002 — Switching backends keeps the user on the same page
      if (conversationMatch) navigate("/conversations");
      else if (automationDetailMatch) navigate("/automations");

      setActive(target.id, orgId);
      onSelectOption?.();
    },
    [
      activeValue,
      backends,
      conversationMatch,
      automationDetailMatch,
      navigate,
      options,
      setActive,
      t,
      onSelectOption,
    ],
  );

  return (
    <>
      <div className="flex items-center gap-2 w-full">
        <div className="flex-1 min-w-0">
          <Dropdown
            testId="backend-selector"
            key={`${activeValue}-${activeOption?.label ?? ""}`}
            defaultValue={
              activeOption ?? {
                value: activeValue,
                label: noBackendSelected ? noBackendLabel : active.backend.name,
                prefix: noBackendSelected
                  ? buildNoBackendPrefix()
                  : buildStatusPrefix(healthByBackendId[active.backend.id]),
              }
            }
            footer={addBackendFooter}
            openUpward={openUpward}
            hideTrigger={hideTrigger}
            defaultOpen={defaultOpen}
            openOnHover={!hideTrigger}
            onChange={(item) => {
              if (!item) return;
              void handleSelectBackend(item.value);
            }}
            placeholder={
              noBackendSelected ? noBackendLabel : active.backend.name
            }
            loading={someCloudLoading}
            options={options}
            className="h-10 px-2 py-0 bg-transparent border-transparent hover:bg-[var(--oh-surface-raised)] focus-within:bg-[var(--oh-surface-raised)] focus-within:border-transparent focus-within:ring-0"
          />
        </div>
        {!hideTrigger ? (
          <StyledTooltip
            content={settingsLabel}
            placement={settingsTooltipPlacement}
            offset={10}
          >
            <NavigationLink
              to="/settings"
              data-testid="backend-selector-settings-link"
              data-active={isSettingsActive}
              aria-label={settingsLabel}
              className={
                isSettingsActive
                  ? cn(
                      "inline-flex items-center justify-center shrink-0 w-9 h-9 rounded-md bg-tertiary text-white font-normal cursor-pointer",
                      formControlTransitionClassName,
                    )
                  : cn(
                      "inline-flex items-center justify-center shrink-0 w-9 h-9 rounded-md text-[var(--oh-muted)] hover:text-white hover:bg-[var(--oh-surface-raised)] cursor-pointer",
                      formControlTransitionClassName,
                    )
              }
            >
              <Settings width={16} height={16} />
            </NavigationLink>
          </StyledTooltip>
        ) : null}
      </div>
      {addBackendModalOpen ? (
        <AddBackendModal onClose={() => setAddBackendModalOpen(false)} />
      ) : null}
      {manageBackendsModalOpen ? (
        <ManageBackendsModal
          onClose={() => setManageBackendsModalOpen(false)}
        />
      ) : null}
    </>
  );
}
