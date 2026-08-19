import React from "react";
import { useTranslation } from "react-i18next";
import { SidebarRailBody } from "./sidebar-rail-body";
import { getErrorStatus, useSettings } from "#/hooks/query/use-settings";
import { useConfig } from "#/hooks/query/use-config";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { cn } from "#/utils/utils";
import { useSidebarMobileNav } from "./sidebar-mobile-nav-context";
import { useSidebarStore } from "#/stores/sidebar-store";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
// The LLM settings modal is only mounted when the settings query 404s and
// LLM settings aren't hidden — keep it out of the sidebar's eager graph.
const SettingsModal = React.lazy(() =>
  import("#/components/shared/modals/settings/settings-modal").then((m) => ({
    default: m.SettingsModal,
  })),
);

// Add/Manage backend modals are lifted into the sidebar (instead of living
// inside BackendSelector) so they survive the collapsed popover unmounting
// when the user moves the cursor out of the popover toward the modal.
const AddBackendModal = React.lazy(() =>
  import("#/components/features/backends/add-backend-modal").then((m) => ({
    default: m.AddBackendModal,
  })),
);
const ManageBackendsModal = React.lazy(() =>
  import("#/components/features/backends/manage-backends-modal").then((m) => ({
    default: m.ManageBackendsModal,
  })),
);

const MOBILE_DRAWER_TRANSITION_MS = 250;

export function Sidebar() {
  const { t } = useTranslation("openhands");
  const { currentPath } = useNavigation();
  const { data: config } = useConfig();
  const {
    data: settings,
    error: settingsError,
    isError: settingsIsError,
    isFetching: isFetchingSettings,
  } = useSettings();
  const { backends, active } = useActiveBackendContext();
  const healthByBackendId = useBackendsHealth(backends);
  const activeBackendHealth = healthByBackendId[active.backend.id];
  const collapsed = useSidebarStore((state) => state.collapsed);
  const setCollapsed = useSidebarStore((state) => state.setCollapsed);
  const [settingsModalIsOpen, setSettingsModalIsOpen] = React.useState(false);
  const [collapsedBackendPopoverOpen, setCollapsedBackendPopoverOpen] =
    React.useState(false);
  const collapsedBackendCloseTimer = React.useRef<ReturnType<
    typeof setTimeout
  > | null>(null);
  // Lifted out of BackendSelector so opening these modals from the
  // collapsed-sidebar popover doesn't lose state when the popover unmounts
  // (cursor moving toward the modal triggers onMouseLeave -> close).
  const [addBackendModalOpen, setAddBackendModalOpen] = React.useState(false);
  const [manageBackendsModalOpen, setManageBackendsModalOpen] =
    React.useState(false);
  const [collapsedRailHovered, setCollapsedRailHovered] = React.useState(false);
  const suppressCollapsedExpandRef = React.useRef(false);
  const [, refreshCollapsedExpandGate] = React.useReducer((n) => n + 1, 0);
  const { isOpen: isMobileNavOpen, close: closeMobileNav } =
    useSidebarMobileNav();
  const [mobileDrawerMounted, setMobileDrawerMounted] = React.useState(false);
  const [mobileDrawerVisible, setMobileDrawerVisible] = React.useState(false);
  const collapsedBackendPopoverRef = useClickOutsideElement<HTMLDivElement>(
    () => setCollapsedBackendPopoverOpen(false),
  );
  const settingsErrorStatus = getErrorStatus(settingsError);

  React.useEffect(() => {
    closeMobileNav();
  }, [currentPath, closeMobileNav]);

  React.useEffect(() => {
    if (isMobileNavOpen) {
      setMobileDrawerMounted(true);
      const frame = requestAnimationFrame(() => {
        setMobileDrawerVisible(true);
      });
      return () => cancelAnimationFrame(frame);
    }

    setMobileDrawerVisible(false);
    const timer = window.setTimeout(() => {
      setMobileDrawerMounted(false);
    }, MOBILE_DRAWER_TRANSITION_MS);
    return () => window.clearTimeout(timer);
  }, [isMobileNavOpen]);

  React.useEffect(() => {
    if (!isMobileNavOpen) {
      return undefined;
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeMobileNav();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isMobileNavOpen, closeMobileNav]);

  React.useEffect(() => {
    if (currentPath === "/settings") {
      setSettingsModalIsOpen(false);
    } else if (
      !isFetchingSettings &&
      settingsIsError &&
      settingsErrorStatus !== 404
    ) {
      // We don't show toast errors for settings in the global error handler
      // because we have a special case for 404 errors
      displayErrorToast(t(I18nKey.SETTINGS$FETCH_ERROR));
    } else if (
      settingsErrorStatus === 404 &&
      !config?.feature_flags?.hide_llm_settings
    ) {
      setSettingsModalIsOpen(true);
    }
  }, [
    currentPath,
    isFetchingSettings,
    settingsIsError,
    settingsErrorStatus,
    config?.feature_flags?.hide_llm_settings,
    t,
  ]);

  const collapseToggleLabel = t(
    collapsed ? I18nKey.SIDEBAR$EXPAND : I18nKey.SIDEBAR$COLLAPSE,
  );
  const handleCollapsedRailClick = React.useCallback(
    (event: React.MouseEvent<HTMLElement>) => {
      if (!collapsed) {
        return;
      }

      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }

      // Keep existing behavior for explicit controls/links and only use
      // this as a convenience hit-area for empty collapsed-rail space.
      if (
        target.closest(
          "a,button,input,textarea,select,[role='button'],[role='link']",
        )
      ) {
        return;
      }

      setCollapsed(false);
    },
    [collapsed, setCollapsed],
  );
  const handleCollapse = React.useCallback(() => {
    setCollapsedRailHovered(false);
    suppressCollapsedExpandRef.current = true;
    refreshCollapsedExpandGate();
    setCollapsed(true);
    window.setTimeout(() => {
      suppressCollapsedExpandRef.current = false;
      refreshCollapsedExpandGate();
    }, 250);
  }, [setCollapsed]);
  const showCollapsedExpandButton =
    collapsed && collapsedRailHovered && !suppressCollapsedExpandRef.current;

  const isExtensionsActive =
    currentPath === "/customize" ||
    currentPath.startsWith("/skills") ||
    currentPath === "/plugins" ||
    currentPath === "/mcp" ||
    currentPath === "/sub-agents";

  const railBodyProps = {
    collapseToggleLabel,
    onCollapse: handleCollapse,
    onExpand: () => setCollapsed(false),
    showCollapsedExpandButton,
    isExtensionsActive,
    currentPath,
    activeBackendHealth,
    collapsedBackendPopoverOpen,
    setCollapsedBackendPopoverOpen,
    collapsedBackendPopoverRef,
    collapsedBackendCloseTimer,
    onOpenAddBackend: () => setAddBackendModalOpen(true),
    onOpenManageBackends: () => setManageBackendsModalOpen(true),
  };

  return (
    <>
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- the aside acts as a hit-area for the collapsed rail; nested controls handle their own keyboard interactions. */}
      <aside
        aria-label={t(I18nKey.SIDEBAR$NAVIGATION_LABEL)}
        data-collapsed={collapsed ? "true" : "false"}
        onClick={handleCollapsedRailClick}
        onMouseEnter={() => {
          if (collapsed) {
            setCollapsedRailHovered(true);
          }
        }}
        onMouseLeave={() => {
          setCollapsedRailHovered(false);
        }}
        className={cn(
          "max-md:hidden flex bg-base flex-col min-h-0 transition-[width,min-width] duration-200",
          "md:border-r md:border-[var(--oh-border)] md:h-full",
          collapsed
            ? "md:w-[60px] md:min-w-[60px] md:px-2.5"
            : "md:w-[300px] md:min-w-[300px] pb-2 md:pl-2.5 md:pr-0",
          currentPath === "/" && "md:pb-3",
        )}
      >
        <SidebarRailBody
          collapsed={collapsed}
          showCollapseToggle
          {...railBodyProps}
        />
      </aside>

      {mobileDrawerMounted ? (
        <>
          <div
            className={cn(
              "fixed inset-0 z-40 bg-black/50 md:hidden",
              "transition-opacity ease-in-out motion-reduce:transition-none",
              mobileDrawerVisible
                ? "opacity-100"
                : "pointer-events-none opacity-0",
            )}
            style={{ transitionDuration: `${MOBILE_DRAWER_TRANSITION_MS}ms` }}
            onClick={closeMobileNav}
            aria-hidden={!mobileDrawerVisible}
          />
          <aside
            aria-label={t(I18nKey.SIDEBAR$NAVIGATION_LABEL)}
            data-testid="sidebar-mobile-drawer"
            aria-hidden={!mobileDrawerVisible}
            className={cn(
              "fixed inset-y-0 left-0 z-50 flex min-h-0 w-[min(300px,85vw)] flex-col bg-base",
              "border-r border-[var(--oh-border)] pb-2 pl-2.5 pr-0 md:hidden",
              "transition-transform ease-in-out motion-reduce:transition-none",
              mobileDrawerVisible ? "translate-x-0" : "-translate-x-full",
            )}
            style={{ transitionDuration: `${MOBILE_DRAWER_TRANSITION_MS}ms` }}
          >
            <SidebarRailBody
              collapsed={false}
              showCollapseToggle={false}
              showMobileCloseButton
              onCloseMobile={closeMobileNav}
              {...railBodyProps}
            />
          </aside>
        </>
      ) : null}

      {settingsModalIsOpen && (
        <React.Suspense fallback={null}>
          <SettingsModal
            settings={settings}
            onClose={() => setSettingsModalIsOpen(false)}
          />
        </React.Suspense>
      )}
      {addBackendModalOpen && (
        <React.Suspense fallback={null}>
          <AddBackendModal onClose={() => setAddBackendModalOpen(false)} />
        </React.Suspense>
      )}
      {manageBackendsModalOpen && (
        <React.Suspense fallback={null}>
          <ManageBackendsModal
            onClose={() => setManageBackendsModalOpen(false)}
          />
        </React.Suspense>
      )}
    </>
  );
}
