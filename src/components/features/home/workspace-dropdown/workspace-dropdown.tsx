import React, { useState, useMemo, useCallback, useRef } from "react";
import { useCombobox } from "downshift";
import { useTranslation } from "react-i18next";

import { cn } from "#/utils/utils";
import {
  dropdownFooterActionClassName,
  dropdownMenuListClassName,
} from "#/utils/dropdown-classes";
import { formControlFieldClassName } from "#/utils/form-control-classes";
import type { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import { I18nKey } from "#/i18n/declaration";
import RepoIcon from "#/icons/repo.svg?react";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";

import { ClearButton } from "../shared/clear-button";
import { ToggleButton } from "../shared/toggle-button";
import { DropdownItem } from "../shared/dropdown-item";
import { EmptyState } from "../shared/empty-state";
import { GenericDropdownMenu } from "../shared/generic-dropdown-menu";

// Sentinel group key for standalone workspaces (no parent). Module-scoped so
// it's a stable reference for hooks and reads clearly as a non-path sentinel.
const STATIC_GROUP_KEY = "__ungrouped__";

export interface WorkspaceDropdownProps {
  workspaces: LocalWorkspace[];
  /**
   * The workspace parents that produced the dynamic children in `workspaces`
   * (from `useResolvedWorkspaces`, including the implicit `/projects`). Used to
   * label each folder's group by its parent's `name`. When two or more distinct
   * groups are present the list renders grouped under headers; with one group
   * (or omitted) it stays flat. A child whose `parentPath` has no matching
   * parent here falls back to the path basename.
   */
  parents?: LocalWorkspaceParent[];
  value: LocalWorkspace | null;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  disabledTooltip?: string | null;
  /**
   * Whether to surface the "Manage Workspaces" entry in the sticky footer.
   * Defaults to `workspaces.length > 0` when omitted; pass an explicit value
   * if there are workspace parents (whose children may not have loaded yet)
   * that should also count as "manageable".
   */
  showManage?: boolean;
  onChange: (workspace: LocalWorkspace | null) => void;
  onAddClick: () => void;
  onManageClick: () => void;
}

export function WorkspaceDropdown({
  workspaces,
  parents = [],
  value,
  placeholder,
  className,
  disabled = false,
  disabledTooltip,
  showManage,
  onChange,
  onAddClick,
  onManageClick,
}: WorkspaceDropdownProps) {
  const { t } = useTranslation("openhands");
  const [inputValue, setInputValue] = useState(value?.name ?? "");
  const menuRef = useRef<HTMLUListElement>(null);

  const filteredWorkspaces = useMemo(() => {
    const trimmed = inputValue.trim().toLowerCase();
    if (!trimmed) return workspaces;
    return workspaces.filter(
      (w) =>
        w.name.toLowerCase().includes(trimmed) ||
        w.path.toLowerCase().includes(trimmed),
    );
  }, [workspaces, inputValue]);

  // Group the filtered list by parent so folders from the same workspace render
  // contiguously under a header. The grouped array is the SINGLE source for both
  // downshift's `items` and the menu render, so `highlightedIndex` and keyboard
  // navigation stay in lockstep with the visible order. Headers are presentational
  // siblings (see `renderItemPrefix`) and consume no downshift index; each option
  // carries its group label as an accessible name so screen-reader users get the
  // grouping the presentational header can't convey.
  const groupKeyOf = useCallback(
    (w: LocalWorkspace) => w.parentPath ?? STATIC_GROUP_KEY,
    [],
  );
  const parentNameByPath = useMemo(() => {
    const map = new Map<string, string>();
    parents.forEach((p) => map.set(p.path, p.name));
    return map;
  }, [parents]);

  const {
    groupedWorkspaces,
    isGrouped,
    headerByFirstIndex,
    groupLabelByIndex,
  } = useMemo(() => {
    const order: string[] = [];
    const byGroup = new Map<string, LocalWorkspace[]>();
    filteredWorkspaces.forEach((w) => {
      const key = groupKeyOf(w);
      if (!byGroup.has(key)) {
        byGroup.set(key, []);
        order.push(key);
      }
      byGroup.get(key)?.push(w);
    });

    // Keep the catch-all "Other" (no-parent) group last, wherever its first
    // member happened to appear. Stable sort preserves named-group order.
    order.sort((a, b) => {
      if (a === STATIC_GROUP_KEY) return 1;
      if (b === STATIC_GROUP_KEY) return -1;
      return 0;
    });

    const grouping = order.length > 1;
    const grouped: LocalWorkspace[] = [];
    const headers = new Map<number, string>();
    const labelByIndex = new Map<number, string>();
    order.forEach((key) => {
      // `||` (not `??`) so an empty parent name falls through to the basename.
      const label =
        key === STATIC_GROUP_KEY
          ? t(I18nKey.HOME$WORKSPACE_GROUP_OTHER)
          : parentNameByPath.get(key) ||
            key.split("/").filter(Boolean).pop() ||
            key;
      if (grouping) {
        headers.set(grouped.length, label);
      }
      (byGroup.get(key) ?? []).forEach((w) => {
        labelByIndex.set(grouped.length, label);
        grouped.push(w);
      });
    });

    return {
      groupedWorkspaces: grouped,
      isGrouped: grouping,
      headerByFirstIndex: headers,
      groupLabelByIndex: labelByIndex,
    };
  }, [filteredWorkspaces, groupKeyOf, parentNameByPath, t]);

  const handleSelectionChange = useCallback(
    (selectedItem: LocalWorkspace | null) => {
      onChange(selectedItem);
      if (selectedItem) {
        setInputValue(selectedItem.name);
      }
    },
    [onChange],
  );

  const handleClear = useCallback(() => {
    handleSelectionChange(null);
    setInputValue("");
  }, [handleSelectionChange]);

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    highlightedIndex,
    getItemProps,
    selectedItem,
    closeMenu,
  } = useCombobox<LocalWorkspace>({
    items: groupedWorkspaces,
    itemToString: (item) => item?.name ?? "",
    selectedItem: value,
    onSelectedItemChange: ({ selectedItem: newSelectedItem }) => {
      handleSelectionChange(newSelectedItem ?? null);
    },
    inputValue,
    onIsOpenChange: ({
      isOpen: newIsOpen,
      selectedItem: currentSelectedItem,
    }) => {
      if (newIsOpen) {
        setInputValue("");
      } else {
        setInputValue(currentSelectedItem?.name ?? "");
      }
    },
    stateReducer: (state, actionAndChanges) =>
      actionAndChanges.type === useCombobox.stateChangeTypes.InputClick &&
      state.isOpen
        ? { ...actionAndChanges.changes, isOpen: true }
        : actionAndChanges.changes,
  });

  const renderItem = (
    item: LocalWorkspace,
    index: number,
    itemHighlightedIndex: number,
    itemSelectedItem: LocalWorkspace | null,
    itemGetItemProps: any, // eslint-disable-line @typescript-eslint/no-explicit-any
  ) => (
    <DropdownItem
      key={item.id}
      item={item}
      index={index}
      isSelected={itemSelectedItem?.id === item.id}
      getItemProps={itemGetItemProps}
      getDisplayText={(workspace) => workspace.name}
      getItemKey={(workspace) => workspace.id}
      ariaLabel={
        isGrouped
          ? `${groupLabelByIndex.get(index) ?? ""}, ${item.name}`.trim()
          : undefined
      }
    />
  );

  const renderEmptyState = (emptyInputValue: string) => (
    <EmptyState
      inputValue={emptyInputValue}
      searchMessage={t(I18nKey.HOME$NO_WORKSPACES)}
      emptyMessage={t(I18nKey.HOME$NO_WORKSPACES)}
      testId="workspace-dropdown-empty"
    />
  );

  const preventDropdownMenuClose = useCallback(
    (event: React.SyntheticEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
    },
    [],
  );

  const handleAddClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      closeMenu();
      onAddClick();
    },
    [closeMenu, onAddClick, preventDropdownMenuClose],
  );

  const handleAddTouchEnd = useCallback(
    (event: React.TouchEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      closeMenu();
      onAddClick();
    },
    [closeMenu, onAddClick, preventDropdownMenuClose],
  );

  const handleManageClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      closeMenu();
      onManageClick();
    },
    [closeMenu, onManageClick, preventDropdownMenuClose],
  );

  const handleManageTouchEnd = useCallback(
    (event: React.TouchEvent<HTMLButtonElement>) => {
      preventDropdownMenuClose(event);
      closeMenu();
      onManageClick();
    },
    [closeMenu, onManageClick, preventDropdownMenuClose],
  );

  const stickyFooterItem = useMemo(
    () => (
      <div className={dropdownMenuListClassName}>
        <button
          type="button"
          data-testid="add-workspaces-button"
          className={cn(dropdownFooterActionClassName, "cursor-pointer")}
          onMouseDown={preventDropdownMenuClose}
          onTouchStart={preventDropdownMenuClose}
          onTouchEnd={handleAddTouchEnd}
          onClick={handleAddClick}
        >
          {t(I18nKey.HOME$ADD_WORKSPACES)}
        </button>
        {(showManage ?? workspaces.length > 0) && (
          <button
            type="button"
            data-testid="manage-workspaces-button"
            className={cn(dropdownFooterActionClassName, "cursor-pointer")}
            onMouseDown={preventDropdownMenuClose}
            onTouchStart={preventDropdownMenuClose}
            onTouchEnd={handleManageTouchEnd}
            onClick={handleManageClick}
          >
            {t(I18nKey.HOME$MANAGE_WORKSPACES)}
          </button>
        )}
      </div>
    ),
    [
      handleAddClick,
      handleAddTouchEnd,
      handleManageClick,
      handleManageTouchEnd,
      preventDropdownMenuClose,
      t,
      workspaces.length,
      showManage,
    ],
  );

  const control = (
    <div className={cn("relative", className)}>
      <div className="group relative text-[var(--oh-muted)] hover:text-white">
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 z-10">
          <RepoIcon width={16} height={16} />
        </div>
        <input
          {...getInputProps({
            disabled,
            placeholder:
              isOpen && value
                ? value.name
                : (placeholder ?? t(I18nKey.HOME$WORKSPACE_PLACEHOLDER)),
            className: cn(
              formControlFieldClassName,
              "text-inherit shadow-none pl-7 pr-16 text-sm font-normal leading-5",
              "placeholder:text-[var(--oh-muted)]",
              "disabled:cursor-not-allowed disabled:opacity-60",
            ),
            onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
              setInputValue(e.target.value);
            },
          })}
          data-testid="workspace-dropdown"
        />

        <div className="absolute right-1 top-1/2 transform -translate-y-1/2 flex items-center">
          {value && <ClearButton disabled={disabled} onClear={handleClear} />}
          <ToggleButton
            isOpen={isOpen}
            disabled={disabled}
            getToggleButtonProps={getToggleButtonProps}
          />
        </div>
      </div>

      <GenericDropdownMenu<LocalWorkspace>
        isOpen={isOpen}
        filteredItems={groupedWorkspaces}
        inputValue={inputValue}
        highlightedIndex={highlightedIndex}
        selectedItem={selectedItem}
        getMenuProps={getMenuProps}
        getItemProps={getItemProps}
        menuRef={menuRef}
        renderItem={renderItem}
        renderItemPrefix={
          isGrouped
            ? (_item, index) => {
                const label = headerByFirstIndex.get(index);
                if (!label) return null;
                return (
                  <li
                    role="presentation"
                    aria-hidden="true"
                    data-testid="workspace-group-header"
                    className="px-2 pt-2 pb-1 text-xs font-medium text-[var(--oh-muted)] select-none"
                  >
                    {label}
                  </li>
                );
              }
            : undefined
        }
        renderEmptyState={renderEmptyState}
        stickyFooterItem={stickyFooterItem}
        testId="workspace-dropdown-menu"
        itemKey={(item) => item.id}
      />
    </div>
  );

  if (!disabledTooltip) {
    return control;
  }

  return (
    <StyledTooltip content={disabledTooltip} placement="top">
      <span className="block">{control}</span>
    </StyledTooltip>
  );
}
