import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import type { BackendKind } from "#/api/backend-registry/types";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { getWorkspaceModeI18nKey } from "#/utils/workspace-mode";
import { cn } from "#/utils/utils";
import {
  dropdownMenuListClassName,
  dropdownMenuRowClassName,
  dropdownMenuRowIconWrapperClassName,
} from "#/utils/dropdown-classes";
import { WorkspaceModeIcon } from "./workspace-mode-icon";

interface WorkspaceModeSelectorProps {
  value: WorkspaceMode;
  backendKind: BackendKind;
  onChange: (value: WorkspaceMode) => void;
  disabled?: boolean;
  testId?: string;
}

const WORKSPACE_MODE_OPTIONS: WorkspaceMode[] = ["local_repo", "new_worktree"];

export function WorkspaceModeSelector({
  value,
  backendKind,
  onChange,
  disabled = false,
  testId = "workspace-mode-selector",
}: WorkspaceModeSelectorProps) {
  const { t } = useTranslation("openhands");
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const label = t(getWorkspaceModeI18nKey(value, backendKind));

  useEffect(() => {
    if (!open) return undefined;

    const onMouseDown = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  return (
    <div ref={containerRef} className="relative inline-flex">
      <button
        type="button"
        data-testid={testId}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        disabled={disabled}
        onClick={() => setOpen((next) => !next)}
        className={cn(
          "group flex flex-row items-center justify-between gap-2 pl-2.5 pr-2 py-1 rounded-[100px] truncate relative",
          "border border-[rgba(71,74,84,0.50)] bg-transparent text-white",
          disabled
            ? "cursor-not-allowed opacity-50"
            : "cursor-pointer hover:border-[var(--oh-border-subtle)]",
        )}
      >
        <span className="flex size-3 shrink-0 items-center justify-center">
          <WorkspaceModeIcon mode={value} />
        </span>
        <span className="min-w-0 flex-1 truncate text-sm font-normal leading-5">
          {label}
        </span>
        <ChevronDown className="size-3 shrink-0 text-white" aria-hidden />
      </button>

      {open ? (
        <div
          role="menu"
          data-testid={`${testId}-menu`}
          className={cn(
            "absolute left-0 top-full z-50 mt-1 min-w-[180px] rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface)] p-1 shadow-lg",
            dropdownMenuListClassName,
          )}
        >
          {WORKSPACE_MODE_OPTIONS.map((option) => {
            const optionLabel = t(getWorkspaceModeI18nKey(option, backendKind));
            return (
              <button
                key={option}
                type="button"
                role="menuitemradio"
                aria-checked={option === value}
                data-testid={`${testId}-option-${option}`}
                onClick={() => {
                  onChange(option);
                  setOpen(false);
                }}
                className={dropdownMenuRowClassName}
              >
                <span className={dropdownMenuRowIconWrapperClassName}>
                  <WorkspaceModeIcon mode={option} />
                </span>
                <span className="truncate">{optionLabel}</span>
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
