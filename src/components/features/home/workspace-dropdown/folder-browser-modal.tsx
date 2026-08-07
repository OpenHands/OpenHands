import React, { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { I18nKey } from "#/i18n/declaration";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import {
  type HomeDirectoryResponse,
  useHomeDirectory,
  useSearchSubdirs,
} from "#/hooks/query/use-search-subdirs";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSettings } from "#/hooks/query/use-settings";
import { useCloneRepository } from "#/hooks/mutation/use-clone-repository";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { cn } from "#/utils/utils";
import { modalTitleSmClassName } from "#/utils/modal-classes";
import FolderIcon from "#/icons/folder.svg?react";
import ChevronLeft from "#/icons/chevron-left-small.svg?react";

const PROJECTS_PATH = "/projects";
const NONE_PROVIDER_KEY = "__none__";

interface FolderBrowserModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (items: LocalWorkspace[]) => void;
  onAddParent?: (items: LocalWorkspaceParent[]) => void;
}

interface SidebarEntry {
  label: string;
  path: string;
}

interface SidebarSectionProps {
  label: string;
  entries: SidebarEntry[];
  currentPath: string | null;
  onPick: (path: string) => void;
}

function SidebarSection({
  label,
  entries,
  currentPath,
  onPick,
}: SidebarSectionProps) {
  if (entries.length === 0) return null;
  return (
    <div className="px-2 pb-3">
      <div className="px-2 pb-1 text-[11px] uppercase tracking-wide text-[var(--oh-muted)] font-semibold">
        {label}
      </div>
      <ul>
        {entries.map((entry) => {
          const isActive = currentPath === entry.path;
          return (
            <li key={entry.path}>
              <button
                type="button"
                onClick={() => onPick(entry.path)}
                data-testid={`folder-browser-sidebar-${entry.label.toLowerCase()}`}
                className={cn(
                  "flex items-center gap-2 w-full px-2 py-1 rounded text-sm cursor-pointer",
                  isActive
                    ? "bg-tertiary text-white"
                    : "text-[var(--oh-text-tertiary)] hover:bg-[var(--oh-surface-raised)]",
                )}
              >
                <FolderIcon width={14} height={14} className="shrink-0" />
                <span className="truncate">{entry.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function getParentPath(path: string): string | null {
  const trimmed = trimTrailingSeparators(path);
  if (!trimmed || trimmed === "/" || isWindowsDriveRoot(trimmed)) return null;

  const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (idx < 0) return null;
  if (idx === 0) return "/";

  const parent = trimmed.slice(0, idx);
  if (/^[A-Za-z]:$/.test(parent)) {
    return `${parent}${trimmed[idx]}`;
  }

  return parent;
}

function isWindowsDriveRoot(path: string): boolean {
  return /^[A-Za-z]:[\\/]?$/.test(path);
}

function trimTrailingSeparators(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  if (/^[A-Za-z]:$/.test(trimmed)) {
    const separator = path.includes("/") && !path.includes("\\") ? "/" : "\\";
    return `${trimmed}${separator}`;
  }
  return trimmed;
}

function shouldDefaultToProjectsPath(
  homeData: HomeDirectoryResponse | undefined,
): boolean {
  return homeData?.home === "/home/openhands";
}

function resolveInitialBrowsePath(
  homeData: HomeDirectoryResponse,
  preferredPath: string | null | undefined,
): string {
  const preferred = preferredPath?.trim();
  if (preferred) return preferred;
  return shouldDefaultToProjectsPath(homeData) ? PROJECTS_PATH : homeData.home;
}

type BrowserMode = "browse" | "clone";

export function FolderBrowserModal({
  isOpen,
  onClose,
  onAdd,
  onAddParent,
}: FolderBrowserModalProps) {
  const { t } = useTranslation("openhands");
  const [currentPath, setCurrentPath] = useState<string | null>(null);
  const [mode, setMode] = useState<BrowserMode>("browse");
  const [cloneUrl, setCloneUrl] = useState("");
  const [providerId, setProviderId] = useState<string>(NONE_PROVIDER_KEY);
  const active = useActiveBackend();
  const isLocalBackend = active.backend.kind === "local";

  const { data: homeData } = useHomeDirectory();
  const { data: settings } = useSettings();
  const { mutate: cloneRepository, isPending: isCloning } =
    useCloneRepository();

  const preferredBrowsePath = settings?.default_workspace_browse_path;
  const gitProviders = settings?.git_providers ?? [];

  // Initialize / reset to preferred path (or home) each time the modal is opened
  useEffect(() => {
    if (isOpen && homeData?.home && currentPath === null) {
      setCurrentPath(resolveInitialBrowsePath(homeData, preferredBrowsePath));
    }
    if (!isOpen) {
      setCurrentPath(null);
      setMode("browse");
      setCloneUrl("");
      setProviderId(NONE_PROVIDER_KEY);
    }
  }, [isOpen, homeData?.home, preferredBrowsePath, currentPath]);

  // A backend switch invalidates the previous path — clear it so the
  // open/close effect can re-seed from the new backend's homeData.home.
  useEffect(() => {
    setCurrentPath(null);
  }, [active.backend.id, active.orgId]);

  const {
    data: listing,
    isLoading,
    isError,
    error,
  } = useSearchSubdirs(isOpen && mode === "browse" ? currentPath : null);

  const favorites: SidebarEntry[] = useMemo(() => {
    if (!homeData?.home) return [];
    const trimmed = trimTrailingSeparators(homeData.home) || homeData.home;
    const backendFavorites = [
      { label: "Home", path: trimmed },
      ...(homeData.favorites ?? []),
    ];
    if (
      shouldDefaultToProjectsPath(homeData) &&
      !backendFavorites.some((entry) => entry.path === PROJECTS_PATH)
    ) {
      backendFavorites.push({
        label: PROJECTS_PATH,
        path: PROJECTS_PATH,
      });
    }
    const preferred = preferredBrowsePath?.trim();
    if (
      preferred &&
      !backendFavorites.some((entry) => entry.path === preferred)
    ) {
      backendFavorites.unshift({
        label: t(I18nKey.HOME$DEFAULT_WORKSPACE_PATH),
        path: preferred,
      });
    }

    return backendFavorites;
  }, [homeData, preferredBrowsePath, t]);

  const locations: SidebarEntry[] = homeData?.locations ?? [];

  const providerItems = useMemo(
    () => [
      {
        key: NONE_PROVIDER_KEY,
        label: t(I18nKey.HOME$CLONE_NO_PROVIDER),
      },
      ...gitProviders.map((provider) => ({
        key: provider.id,
        label: provider.label,
      })),
    ],
    [gitProviders, t],
  );

  if (!isOpen) return null;

  const subdirs = listing?.items ?? [];
  const parent = currentPath ? getParentPath(currentPath) : null;

  const showHostHomeHint =
    homeData?.home === "/home/openhands" &&
    (homeData?.favorites?.length ?? 0) === 0 &&
    currentPath === homeData?.home &&
    !isLoading &&
    !isError &&
    subdirs.length === 0;

  const getBasename = (path: string): string => {
    const trimmed = trimTrailingSeparators(path);
    if (!trimmed) return "/";
    if (trimmed === "/" || isWindowsDriveRoot(trimmed)) return trimmed;
    const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
    return idx >= 0 ? trimmed.slice(idx + 1) || trimmed : trimmed;
  };

  const handleAddDirectory = () => {
    if (!currentPath) return;
    const item: LocalWorkspace = {
      id: currentPath,
      name: getBasename(currentPath),
      path: currentPath,
    };
    onAdd([item]);
    onClose();
  };

  const handleAddAllSubdirectories = () => {
    if (!currentPath || !onAddParent) return;
    onAddParent([
      {
        id: currentPath,
        name: getBasename(currentPath),
        path: currentPath,
      },
    ]);
    onClose();
  };

  const resolveCloneParentPath = (): string | null => {
    const preferred = preferredBrowsePath?.trim();
    if (preferred) return preferred;
    if (homeData?.home) {
      return shouldDefaultToProjectsPath(homeData)
        ? PROJECTS_PATH
        : homeData.home;
    }
    return null;
  };

  const handleClone = () => {
    const url = cloneUrl.trim();
    const parentPath = resolveCloneParentPath();
    if (!url) {
      displayErrorToast(t(I18nKey.HOME$CLONE_URL_REQUIRED));
      return;
    }
    if (!parentPath) {
      displayErrorToast(t(I18nKey.HOME$CLONE_PARENT_REQUIRED));
      return;
    }

    cloneRepository(
      {
        url,
        parentPath,
        ...(providerId !== NONE_PROVIDER_KEY ? { providerId } : {}),
      },
      {
        onSuccess: (result) => {
          onAdd([
            {
              id: result.path,
              name: result.name,
              path: result.path,
            },
          ]);
          displaySuccessToast(
            t(I18nKey.HOME$CLONE_SUCCESS, { name: result.name }),
          );
          onClose();
        },
        onError: (cloneError) => {
          if (isSdkHttpStatusError(cloneError, 404)) {
            displayErrorToast(t(I18nKey.HOME$CLONE_UNSUPPORTED));
            return;
          }
          displayErrorToast(
            retrieveAxiosErrorMessage(cloneError) ||
              t(I18nKey.HOME$CLONE_FAILED),
          );
        },
      },
    );
  };

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.HOME$ADD_WORKSPACES_TITLE)}
    >
      <div
        data-testid="folder-browser-modal"
        className={cn(
          "flex flex-col bg-[var(--oh-surface)] border border-[var(--oh-border-input)] rounded-xl",
          modalWidthClassName("xl"),
          MODAL_MAX_WIDTH_VIEWPORT,
          "h-[480px]",
        )}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--oh-border-input)]">
          <BaseModalTitle
            className={modalTitleSmClassName}
            title={t(I18nKey.HOME$ADD_WORKSPACES_TITLE)}
          />
        </div>

        {isLocalBackend && (
          <div
            className="flex gap-1 px-5 pt-3"
            data-testid="folder-browser-mode-tabs"
          >
            <BrandButton
              type="button"
              variant={mode === "browse" ? "primary" : "secondary"}
              testId="folder-browser-mode-browse"
              onClick={() => setMode("browse")}
            >
              {t(I18nKey.HOME$BROWSE_FOLDERS)}
            </BrandButton>
            <BrandButton
              type="button"
              variant={mode === "clone" ? "primary" : "secondary"}
              testId="folder-browser-mode-clone"
              onClick={() => setMode("clone")}
            >
              {t(I18nKey.HOME$CLONE_REPOSITORY)}
            </BrandButton>
          </div>
        )}

        {mode === "clone" && isLocalBackend ? (
          <div className="flex flex-1 min-h-0 flex-col gap-4 px-5 py-4">
            <SettingsInput
              testId="clone-repo-url-input"
              label={t(I18nKey.HOME$CLONE_REPO_URL)}
              type="url"
              value={cloneUrl}
              onChange={setCloneUrl}
              placeholder={t(I18nKey.HOME$CLONE_REPO_URL_PLACEHOLDER)}
              className="w-full min-w-0"
            />
            <SettingsDropdownInput
              testId="clone-provider-input"
              name="clone-provider-input"
              label={t(I18nKey.HOME$CLONE_PROVIDER)}
              items={providerItems}
              selectedKey={providerId}
              onSelectionChange={(key) => {
                setProviderId(key?.toString() || NONE_PROVIDER_KEY);
              }}
            />
            <p className="text-xs text-[var(--oh-muted)]">
              {t(I18nKey.HOME$CLONE_DESTINATION_HINT, {
                path: resolveCloneParentPath() ?? "—",
              })}
            </p>
            <div className="mt-auto flex items-center justify-end gap-2">
              <BrandButton
                type="button"
                variant="secondary"
                onClick={onClose}
                testId="folder-browser-cancel"
              >
                {t(I18nKey.HOME$CANCEL)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="primary"
                onClick={handleClone}
                isDisabled={isCloning || !cloneUrl.trim()}
                testId="clone-repository-button"
              >
                {isCloning
                  ? t(I18nKey.HOME$CLONING)
                  : t(I18nKey.HOME$CLONE_REPOSITORY)}
              </BrandButton>
            </div>
          </div>
        ) : (
          <>
            <div className="flex flex-1 min-h-0">
              <aside
                data-testid="folder-browser-sidebar"
                className="w-[180px] shrink-0 border-r border-[var(--oh-border-input)] bg-[var(--oh-surface)] py-3 overflow-y-auto"
              >
                <SidebarSection
                  label={t(I18nKey.HOME$FAVORITES)}
                  entries={favorites}
                  currentPath={currentPath}
                  onPick={setCurrentPath}
                />
                <SidebarSection
                  label={t(I18nKey.HOME$LOCATIONS)}
                  entries={locations}
                  currentPath={currentPath}
                  onPick={setCurrentPath}
                />
              </aside>

              <div className="flex-1 flex flex-col min-w-0">
                <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--oh-border-input)]">
                  <button
                    type="button"
                    data-testid="folder-browser-up"
                    onClick={() => parent && setCurrentPath(parent)}
                    disabled={!parent}
                    aria-label={t(I18nKey.COMMON$UP)}
                    className="p-1 rounded hover:bg-[var(--oh-interactive-hover)] text-white disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                  >
                    <ChevronLeft width={16} height={16} />
                  </button>
                  <span
                    className="text-xs text-[var(--oh-muted)] truncate"
                    data-testid="folder-browser-current-path"
                  >
                    {currentPath ?? ""}
                  </span>
                </div>

                <div className="grid grid-cols-[1fr_120px] px-4 py-1 border-b border-[var(--oh-border-input)] text-xs text-[var(--oh-text-secondary)] font-semibold">
                  <span>{t(I18nKey.HOME$NAME)}</span>
                  <span>{t(I18nKey.HOME$KIND)}</span>
                </div>

                <ul
                  className="flex-1 overflow-auto custom-scrollbar-always"
                  data-testid="folder-browser-list"
                >
                  {isLoading && (
                    <li className="px-4 py-2 text-sm text-[var(--oh-text-secondary)]">
                      {t(I18nKey.HOME$LOADING)}
                    </li>
                  )}
                  {isError && (
                    <li
                      className="px-4 py-2 text-sm text-red-400"
                      data-testid="folder-browser-error"
                    >
                      {(error as Error | undefined)?.message ??
                        t(I18nKey.COMMON$FAILED_TO_LOAD)}
                    </li>
                  )}
                  {!isLoading && !isError && subdirs.length === 0 && (
                    <li
                      className="px-4 py-2 text-sm text-[var(--oh-text-secondary)]"
                      data-testid={
                        showHostHomeHint
                          ? "folder-browser-host-home-hint"
                          : "folder-browser-empty"
                      }
                    >
                      {showHostHomeHint
                        ? t(I18nKey.HOME$HOST_HOME_NOT_MOUNTED_HINT)
                        : t(I18nKey.HOME$NO_WORKSPACES)}
                    </li>
                  )}
                  {subdirs.map((entry) => (
                    <li key={entry.path}>
                      <button
                        type="button"
                        onClick={() => setCurrentPath(entry.path)}
                        className="grid grid-cols-[1fr_120px] items-center w-full text-left px-4 py-1.5 text-sm text-white hover:bg-[var(--oh-interactive-hover)] cursor-pointer"
                        data-testid={`folder-browser-entry-${entry.name}`}
                      >
                        <span className="flex items-center gap-2 min-w-0">
                          <FolderIcon
                            width={16}
                            height={16}
                            className="shrink-0"
                          />
                          <span className="truncate">{entry.name}</span>
                        </span>
                        <span className="text-[var(--oh-text-secondary)] text-xs">
                          {t(I18nKey.HOME$FOLDER)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-[var(--oh-border-input)]">
              <BrandButton
                type="button"
                variant="secondary"
                onClick={onClose}
                testId="folder-browser-cancel"
              >
                {t(I18nKey.HOME$CANCEL)}
              </BrandButton>
              {onAddParent && (
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={handleAddAllSubdirectories}
                  isDisabled={!currentPath || isLoading}
                  testId="folder-browser-add-all-subdirs"
                >
                  {t(I18nKey.HOME$ADD_ALL_SUBDIRECTORIES)}
                </BrandButton>
              )}
              <BrandButton
                type="button"
                variant="primary"
                onClick={handleAddDirectory}
                isDisabled={!currentPath || isLoading}
                testId="folder-browser-use"
              >
                {t(I18nKey.HOME$ADD_THIS_DIRECTORY)}
              </BrandButton>
            </div>
          </>
        )}
      </div>
    </ModalBackdrop>
  );
}

/** @internal Exported for unit tests */
export { resolveInitialBrowsePath };
